# loss function library.
# author: ynie
# date: Feb, 2020
import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.data_config import cls_reg_ratio
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
from models.registers import LOSSES
from net_utils.libs import get_layout_bdb_sunrgbd, get_bdb_form_from_corners, \
    recover_points_to_world_sys, get_rotation_matix_result, get_bdb_3d_result, \
    get_bdb_2d_result, physical_violation
dist_chamfer = ChamferDistance()

cls_criterion = nn.CrossEntropyLoss(reduction='mean')
reg_criterion = nn.SmoothL1Loss(reduction='mean')
mse_criterion = nn.MSELoss(reduction='mean')
binary_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')


def cls_reg_loss(cls_result, cls_gt, reg_result, reg_gt):
    cls_loss = cls_criterion(cls_result, cls_gt)
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1)))
    else:
        reg_result = torch.gather(reg_result, 1, cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1))
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion(reg_result, reg_gt)
    return cls_loss, cls_reg_ratio * reg_loss

def get_point_loss(points_in_world_sys, cam_R, cam_K, depth_maps, bdb3D_form, split, obj_masks, mask_status):
    '''
    get the depth loss for each mesh.
    :param points_in_world_sys: Number_of_objects x Number_of_points x 3
    :param cam_R: Number_of_scenes x 3 x 3
    :param cam_K: Number_of_scenes x 3 x 3
    :param depth_maps: Number_of_scenes depth maps in a list
    :param split: Number_of_scenes x 2 matrix
    :return: depth loss
    '''
    depth_loss = 0.
    n_objects = 0
    masked_object_id = -1

    device = points_in_world_sys.device

    for scene_id, obj_interval in enumerate(split):
        # map depth to 3d points in camera system.
        u, v = torch.meshgrid(torch.arange(0, depth_maps[scene_id].size(1)), torch.arange(0, depth_maps[scene_id].size(0)))
        u = u.t().to(device)
        v = v.t().to(device)
        u = u.reshape(-1)
        v = v.reshape(-1)
        z_cam = depth_maps[scene_id][v, u]
        u = u.float()
        v = v.float()

        # non_zero_indices = torch.nonzero(z_cam).t()[0]
        # z_cam = z_cam[non_zero_indices]
        # u = u[non_zero_indices]
        # v = v[non_zero_indices]

        # calculate coordinates
        x_cam = (u - cam_K[scene_id][0][2])*z_cam/cam_K[scene_id][0][0]
        y_cam = (v - cam_K[scene_id][1][2])*z_cam/cam_K[scene_id][1][1]

        # transform to toward-up-right coordinate system
        points_world = torch.cat([z_cam.unsqueeze(-1), -y_cam.unsqueeze(-1), x_cam.unsqueeze(-1)], -1)
        # transform from camera system to world system
        points_world = torch.mm(points_world, cam_R[scene_id].t())

        n_columns = depth_maps[scene_id].size(1)

        for loc_id, obj_id in enumerate(range(*obj_interval)):
            if mask_status[obj_id] == 0:
                continue
            masked_object_id += 1

            bdb2d = obj_masks[scene_id][loc_id]['msk_bdb']
            obj_msk = obj_masks[scene_id][loc_id]['msk']

            u_s, v_s = torch.meshgrid(torch.arange(bdb2d[0], bdb2d[2] + 1), torch.arange(bdb2d[1], bdb2d[3] + 1))
            u_s = u_s.t().long()
            v_s = v_s.t().long()
            index_dep = u_s + n_columns * v_s
            index_dep = index_dep.reshape(-1)
            in_object_indices = obj_msk.reshape(-1).nonzero()[0]

            # remove holes in depth maps
            if len(in_object_indices) == 0:
                continue

            object_pnts = points_world[index_dep,:][in_object_indices,:]
            # remove noisy points that out of bounding boxes
            inner_idx = torch.sum(torch.abs(
                torch.mm(object_pnts - bdb3D_form['centroid'][masked_object_id].view(1, 3), bdb3D_form['basis'][masked_object_id].t())) >
                                  bdb3D_form['coeffs'][masked_object_id], dim=1)

            inner_idx = torch.nonzero(inner_idx == 0).t()[0]

            if inner_idx.nelement() == 0:
                continue

            object_pnts = object_pnts[inner_idx, :]

            dist_1 = dist_chamfer(object_pnts.unsqueeze(0), points_in_world_sys[masked_object_id].unsqueeze(0))[0]
            depth_loss += torch.mean(dist_1)
            n_objects += 1
    return depth_loss/n_objects if n_objects > 0 else torch.tensor(0.).to(device)

class BaseLoss(object):
    '''base loss class'''
    def __init__(self, weight=1):
        '''initialize loss module'''
        self.weight = weight

@LOSSES.register_module
class Null(BaseLoss):
    '''This loss function is for modules where a loss preliminary calculated.'''
    def __call__(self, loss):
        return self.weight * torch.mean(loss)

@LOSSES.register_module
class SVRLoss(BaseLoss):
    def __call__(self, est_data, gt_data, subnetworks, face_sampling_rate):
        device = est_data['mesh_coordinates_results'][0].device
        # chamfer losses
        chamfer_loss = torch.tensor(0.).to(device)
        edge_loss = torch.tensor(0.).to(device)
        boundary_loss = torch.tensor(0.).to(device)

        for stage_id, mesh_coordinates_result in enumerate(est_data['mesh_coordinates_results']):
            mesh_coordinates_result = mesh_coordinates_result.transpose(1, 2)
            # points to points chamfer loss
            dist1, dist2 = dist_chamfer(gt_data['mesh_points'], mesh_coordinates_result)[:2]
            chamfer_loss += (torch.mean(dist1)) + (torch.mean(dist2))

            # boundary loss
            if stage_id == subnetworks - 1:
                if 1 in est_data['boundary_point_ids']:
                    boundary_loss = torch.mean(dist2[est_data['boundary_point_ids']])

            # edge loss
            edge_vec = torch.gather(mesh_coordinates_result, 1,
                                    (est_data['output_edges'][:, :, 0] - 1).unsqueeze(-1).expand(est_data['output_edges'].size(0),
                                                                                     est_data['output_edges'].size(1), 3)) \
                       - torch.gather(mesh_coordinates_result, 1,
                                      (est_data['output_edges'][:, :, 1] - 1).unsqueeze(-1).expand(est_data['output_edges'].size(0),
                                                                                       est_data['output_edges'].size(1), 3))

            edge_vec = edge_vec.view(edge_vec.size(0) * edge_vec.size(1), edge_vec.size(2))
            edge_loss += torch.mean(torch.pow(torch.norm(edge_vec, p=2, dim=1), 2))

        chamfer_loss = 100 * chamfer_loss / len(est_data['mesh_coordinates_results'])
        edge_loss = 100 * edge_loss / len(est_data['mesh_coordinates_results'])
        boundary_loss = 100 * boundary_loss

        # face distance losses
        face_loss = torch.tensor(0.).to(device)
        for points_from_edges_by_step, points_indicator_by_step in zip(est_data['points_from_edges'], est_data['point_indicators']):
            points_from_edges_by_step = points_from_edges_by_step.transpose(1, 2).contiguous()
            _, dist2_face, _, idx2 = dist_chamfer(gt_data['mesh_points'], points_from_edges_by_step)
            idx2 = idx2.long()
            dist2_face = dist2_face.view(dist2_face.shape[0], dist2_face.shape[1] // face_sampling_rate,
                                         face_sampling_rate)

            # average distance to nearest face.
            dist2_face = torch.mean(dist2_face, dim=2)
            local_dens = gt_data['densities'][:, idx2[:]][range(gt_data['densities'].size(0)), range(gt_data['densities'].size(0)), :]
            in_mesh = (dist2_face <= local_dens).float()
            face_loss += binary_cls_criterion(points_indicator_by_step, in_mesh)

        if est_data['points_from_edges']:
            face_loss = face_loss / len(est_data['points_from_edges'])

        return {'chamfer_loss': chamfer_loss, 'face_loss': 0.01 * face_loss,
                'edge_loss': 0.1 * edge_loss, 'boundary_loss': 0.5 * boundary_loss}

@LOSSES.register_module
class PoseLoss(BaseLoss):
    def __call__(self, est_data, gt_data, bins_tensor):
        pitch_cls_loss, pitch_reg_loss = cls_reg_loss(est_data['pitch_cls_result'], gt_data['pitch_cls'], est_data['pitch_reg_result'], gt_data['pitch_reg'])
        roll_cls_loss, roll_reg_loss = cls_reg_loss(est_data['roll_cls_result'], gt_data['roll_cls'], est_data['roll_reg_result'], gt_data['roll_reg'])
        lo_ori_cls_loss, lo_ori_reg_loss = cls_reg_loss(est_data['lo_ori_cls_result'], gt_data['lo_ori_cls'], est_data['lo_ori_reg_result'], gt_data['lo_ori_reg'])
        lo_centroid_loss = reg_criterion(est_data['lo_centroid_result'], gt_data['lo_centroid']) * cls_reg_ratio
        lo_coeffs_loss = reg_criterion(est_data['lo_coeffs_result'], gt_data['lo_coeffs']) * cls_reg_ratio

        lo_bdb3D_result = get_layout_bdb_sunrgbd(bins_tensor, est_data['lo_ori_reg_result'], gt_data['lo_ori_cls'], est_data['lo_centroid_result'],
                                                 est_data['lo_coeffs_result'])
        # layout bounding box corner loss
        lo_corner_loss = cls_reg_ratio * reg_criterion(lo_bdb3D_result, gt_data['lo_bdb3D'])

        return {'pitch_cls_loss':pitch_cls_loss, 'pitch_reg_loss':pitch_reg_loss,
                'roll_cls_loss':roll_cls_loss, 'roll_reg_loss':roll_reg_loss,
                'lo_ori_cls_loss':lo_ori_cls_loss, 'lo_ori_reg_loss':lo_ori_reg_loss,
                'lo_centroid_loss':lo_centroid_loss, 'lo_coeffs_loss':lo_coeffs_loss,
                'lo_corner_loss':lo_corner_loss}, {'lo_bdb3D_result':lo_bdb3D_result}

@LOSSES.register_module
class DetLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        # calculate loss
        size_reg_loss = reg_criterion(est_data['size_reg_result'], gt_data['size_reg']) * cls_reg_ratio
        ori_cls_loss, ori_reg_loss = cls_reg_loss(est_data['ori_cls_result'], gt_data['ori_cls'], est_data['ori_reg_result'], gt_data['ori_reg'])
        centroid_cls_loss, centroid_reg_loss = cls_reg_loss(est_data['centroid_cls_result'], gt_data['centroid_cls'],
                                                          est_data['centroid_reg_result'], gt_data['centroid_reg'])
        offset_2D_loss = reg_criterion(est_data['offset_2D_result'], gt_data['offset_2D'])

        return {'size_reg_loss':size_reg_loss, 'ori_cls_loss':ori_cls_loss, 'ori_reg_loss':ori_reg_loss,
                'centroid_cls_loss':centroid_cls_loss, 'centroid_reg_loss':centroid_reg_loss,
                'offset_2D_loss':offset_2D_loss}

@LOSSES.register_module
class ReconLoss(BaseLoss):
    def __call__(self, est_data, gt_data, extra_results):
        if gt_data['mask_flag'] == 0:
            point_loss = 0.
        else:
            # get the world coordinates for each 3d object.
            bdb3D_form = get_bdb_form_from_corners(extra_results['bdb3D_result'], gt_data['mask_status'])
            obj_points_in_world_sys = recover_points_to_world_sys(bdb3D_form, est_data['meshes'])
            point_loss = 100 * get_point_loss(obj_points_in_world_sys, extra_results['cam_R_result'],
                                              gt_data['K'], gt_data['depth_maps'], bdb3D_form, gt_data['split'],
                                              gt_data['obj_masks'], gt_data['mask_status'])

            # remove samples without depth map
            if torch.isnan(point_loss):
                point_loss = 0.

        return {'mesh_loss':point_loss}

@LOSSES.register_module
class JointLoss(BaseLoss):
    def __call__(self, est_data, gt_data, bins_tensor, layout_results):
        # predicted camera rotation
        cam_R_result = get_rotation_matix_result(bins_tensor,
                                                 gt_data['pitch_cls'], est_data['pitch_reg_result'],
                                                 gt_data['roll_cls'], est_data['roll_reg_result'])

        # projected center
        P_result = torch.stack(
            ((gt_data['bdb2D_pos'][:, 0] + gt_data['bdb2D_pos'][:, 2]) / 2 - (gt_data['bdb2D_pos'][:, 2] - gt_data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
             (gt_data['bdb2D_pos'][:, 1] + gt_data['bdb2D_pos'][:, 3]) / 2 - (gt_data['bdb2D_pos'][:, 3] - gt_data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:, 1]), 1)

        # retrieved 3D bounding box
        bdb3D_result, _ = get_bdb_3d_result(bins_tensor,
                                            gt_data['ori_cls'],
                                            est_data['ori_reg_result'],
                                            gt_data['centroid_cls'],
                                            est_data['centroid_reg_result'],
                                            gt_data['size_cls'],
                                            est_data['size_reg_result'],
                                            P_result,
                                            gt_data['K'],
                                            cam_R_result,
                                            gt_data['split'])

        # 3D bounding box corner loss
        corner_loss = 5 * cls_reg_ratio * reg_criterion(bdb3D_result, gt_data['bdb3D'])

        # 2D bdb loss
        bdb2D_result = get_bdb_2d_result(bdb3D_result, cam_R_result, gt_data['K'], gt_data['split'])
        bdb2D_loss = 20 * cls_reg_ratio * reg_criterion(bdb2D_result, gt_data['bdb2D_from_3D_gt'])

        # physical violation loss
        phy_violation, phy_gt = physical_violation(layout_results['lo_bdb3D_result'], bdb3D_result, gt_data['split'])
        phy_loss = 20 * mse_criterion(phy_violation, phy_gt)

        return {'phy_loss':phy_loss, 'bdb2D_loss':bdb2D_loss, 'corner_loss':corner_loss},\
               {'cam_R_result':cam_R_result, 'bdb3D_result':bdb3D_result}

