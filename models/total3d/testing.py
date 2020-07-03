# Tester for Total3D
# author: ynie
# date: April, 2020
import os
from models.testing import BaseTester
from .training import Trainer
import torch
from net_utils.libs import get_rotation_matrix_gt
from configs.data_config import NYU37_TO_PIX3D_CLS_MAPPING
from models.eval_metrics import get_iou_cuboid
from net_utils.libs import get_layout_bdb_sunrgbd, get_rotation_matix_result, get_bdb_evaluation, get_bdb_2d_result, get_corners_of_bb3d_no_index, get_iou
from scipy.io import savemat
from libs.tools import write_obj
import numpy as np


class Tester(BaseTester, Trainer):
    '''
    Tester object for SCNet.
    '''
    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)

    def to_device(self, data):
        device = self.device

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            '''calculate loss from camera and layout estimation'''
            image = data['image'].to(device)
            pitch_reg = data['camera']['pitch_reg'].float().to(device)
            pitch_cls = data['camera']['pitch_cls'].long().to(device)
            roll_reg = data['camera']['roll_reg'].float().to(device)
            roll_cls = data['camera']['roll_cls'].long().to(device)
            lo_ori_reg = data['layout']['ori_reg'].float().to(device)
            lo_ori_cls = data['layout']['ori_cls'].long().to(device)
            lo_centroid = data['layout']['centroid_reg'].float().to(device)
            lo_coeffs = data['layout']['coeffs_reg'].float().to(device)
            lo_bdb3D = data['layout']['bdb3D'].float().to(device)

            layout_input = {'image':image, 'pitch_reg':pitch_reg, 'pitch_cls':pitch_cls, 'roll_reg':roll_reg,
                            'roll_cls':roll_cls, 'lo_ori_reg':lo_ori_reg, 'lo_ori_cls':lo_ori_cls, 'lo_centroid':lo_centroid,
                            'lo_coeffs':lo_coeffs, 'lo_bdb3D':lo_bdb3D}

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['object_detection', 'joint']:
            '''calculate loss from object instances'''
            # from instance images predict object size, orientation, centroid, 2D projection offset.
            patch = data['boxes_batch']['patch'].to(device)
            g_features = data['boxes_batch']['g_feature'].float().to(device)
            size_reg = data['boxes_batch']['size_reg'].float().to(device)
            size_cls = data['boxes_batch']['size_cls'].float().to(device)
            ori_reg = data['boxes_batch']['ori_reg'].float().to(device)
            ori_cls = data['boxes_batch']['ori_cls'].long().to(device)
            centroid_reg = data['boxes_batch']['centroid_reg'].float().to(device)
            centroid_cls = data['boxes_batch']['centroid_cls'].long().to(device)
            offset_2D = data['boxes_batch']['delta_2D'].float().to(device)
            split = data['obj_split']
            # split of relational pairs for batch learning.
            rel_pair_counts = torch.cat([torch.tensor([0]), torch.cumsum(
                torch.pow(data['obj_split'][:, 1] - data['obj_split'][:, 0], 2), 0)], 0)

            object_input = {'patch':patch, 'g_features':g_features, 'size_reg':size_reg, 'size_cls':size_cls,
                            'ori_reg':ori_reg, 'ori_cls':ori_cls, 'centroid_reg':centroid_reg, 'centroid_cls':centroid_cls,
                            'offset_2D':offset_2D, 'split':split, 'rel_pair_counts':rel_pair_counts}

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['joint']:
            '''calculate mesh loss from object instances'''
            # for depth loss
            bdb3D = data['boxes_batch']['bdb3D'].float().to(device)
            K = data['camera']['K'].float().to(device)
            depth_maps = [depth.float().to(device) for depth in data['depth']]

            # ground-truth camera rotation
            cam_R_gt = get_rotation_matrix_gt(self.cfg.bins_tensor,
                                              pitch_cls, pitch_reg,
                                              roll_cls, roll_reg)

            # Notice: we should conclude the NYU37 classes into pix3d (9) classes before feeding into the network.
            cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
            cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                                torch.argmax(size_cls, dim=1)]] = 1

            '''calculate loss from the interelationship between object and layout.'''
            bdb2D_from_3D_gt = data['boxes_batch']['bdb2D_from_3D'].float().to(device)
            bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

            joint_input = {'bdb3D':bdb3D, 'K':K, 'depth_maps':depth_maps, 'cam_R_gt':cam_R_gt,
                           'cls_codes':cls_codes, 'bdb2D_from_3D_gt':bdb2D_from_3D_gt, 'bdb2D_pos':bdb2D_pos}

        '''output data'''
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'layout_estimation':
            data_output =  layout_input
        elif self.cfg.config[self.cfg.config['mode']]['phase'] == 'object_detection':
            data_output =  object_input
        elif self.cfg.config[self.cfg.config['mode']]['phase'] == 'joint':
            data_output = {**layout_input, **object_input, **joint_input}
        else:
            raise NotImplementedError

        return {**data_output, 'sequence_id': data['sequence_id']}


    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        # Layout IoU
        lo_bdb3D_out = get_layout_bdb_sunrgbd(self.cfg.bins_tensor, est_data['lo_ori_reg_result'],
                                              torch.argmax(est_data['lo_ori_cls_result'], 1), est_data['lo_centroid_result'],
                                              est_data['lo_coeffs_result'])

        layout_iou = []
        for index, sequence_id in enumerate(gt_data['sequence_id']):
            lo_iou = get_iou_cuboid(lo_bdb3D_out[index, :, :].cpu().numpy(), gt_data['lo_bdb3D'][index, :, :].cpu().numpy())
            layout_iou.append(lo_iou)

        # camera orientation for evaluation
        cam_R_out = get_rotation_matix_result(self.cfg.bins_tensor,
                                              torch.argmax(est_data['pitch_cls_result'], 1), est_data['pitch_reg_result'],
                                              torch.argmax(est_data['roll_cls_result'], 1), est_data['roll_reg_result'])

        # projected center
        P_result = torch.stack(((gt_data['bdb2D_pos'][:, 0] + gt_data['bdb2D_pos'][:, 2]) / 2 - (
                gt_data['bdb2D_pos'][:, 2] - gt_data['bdb2D_pos'][:, 0]) * est_data['offset_2D_result'][:, 0],
                                (gt_data['bdb2D_pos'][:, 1] + gt_data['bdb2D_pos'][:, 3]) / 2 - (
                                        gt_data['bdb2D_pos'][:, 3] - gt_data['bdb2D_pos'][:, 1]) * est_data['offset_2D_result'][:, 1]), 1)

        bdb3D_out_form_cpu, bdb3D_out = get_bdb_evaluation(self.cfg.bins_tensor, torch.argmax(est_data['ori_cls_result'], 1), est_data['ori_reg_result'],
                                                       torch.argmax(est_data['centroid_cls_result'], 1), est_data['centroid_reg_result'],
                                                       gt_data['size_cls'], est_data['size_reg_result'], P_result, gt_data['K'], cam_R_out, gt_data['split'], return_bdb=True)

        bdb2D_out = get_bdb_2d_result(bdb3D_out, cam_R_out, gt_data['K'], gt_data['split'])

        nyu40class_ids = []
        IoU3D = []
        IoU2D = []
        for index, evaluate_bdb in enumerate(bdb3D_out_form_cpu):
            NYU40CLASS_ID = int(evaluate_bdb['classid'])
            iou_3D = get_iou_cuboid(get_corners_of_bb3d_no_index(evaluate_bdb['basis'], evaluate_bdb['coeffs'],
                                                                 evaluate_bdb['centroid']),
                                    gt_data['bdb3D'][index, :, :].cpu().numpy())

            box1 = bdb2D_out[index, :].cpu().numpy()
            box2 = gt_data['bdb2D_from_3D_gt'][index, :].cpu().numpy()

            box1 = {'u1': box1[0], 'v1': box1[1], 'u2': box1[2], 'v2': box1[3]}
            box2 = {'u1': box2[0], 'v1': box2[1], 'u2': box2[2], 'v2': box2[3]}

            iou_2D = get_iou(box1, box2)

            nyu40class_ids.append(NYU40CLASS_ID)
            IoU3D.append(iou_3D)
            IoU2D.append(iou_2D)

        '''Save results'''
        if self.cfg.config['log']['save_results']:

            save_path = self.cfg.config['log']['vis_path']

            for index, sequence_id in enumerate(gt_data['sequence_id']):
                save_path_per_img = os.path.join(save_path, str(sequence_id.item()))
                if not os.path.exists(save_path_per_img):
                    os.mkdir(save_path_per_img)

                # save layout results
                savemat(os.path.join(save_path_per_img, 'layout.mat'),
                        mdict={'layout': lo_bdb3D_out[index, :, :].cpu().numpy()})

                # save bounding boxes and camera poses
                interval = gt_data['split'][index].cpu().tolist()
                current_cls = nyu40class_ids[interval[0]:interval[1]]

                savemat(os.path.join(save_path_per_img, 'bdb_3d.mat'),
                        mdict={'bdb': bdb3D_out_form_cpu[interval[0]:interval[1]], 'class_id': current_cls})
                savemat(os.path.join(save_path_per_img, 'r_ex.mat'),
                        mdict={'cam_R': cam_R_out[index, :, :].cpu().numpy()})

                current_faces = est_data['out_faces'][interval[0]:interval[1]].cpu().numpy()
                current_coordinates = est_data['meshes'].transpose(1,2)[interval[0]:interval[1]].cpu().numpy()

                for obj_id, obj_cls in enumerate(current_cls):
                    file_path = os.path.join(save_path_per_img, '%s_%s.obj' % (obj_id, obj_cls))

                    mesh_obj = {'v': current_coordinates[obj_id],
                                'f': current_faces[obj_id]}

                    write_obj(file_path, mesh_obj)

        metrics = {}
        metrics['layout_iou'] = np.mean(layout_iou)
        metrics['iou_3d'] = IoU3D
        metrics['iou_2d'] = IoU2D
        return metrics

    def test_step(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        loss = self.get_metric_values(est_data, data)
        return loss

    def visualize_step(self, epoch, phase, iter, data):
        ''' Performs a visualization step.
        '''
        pass
