# Trainer for Total3D.
# author: ynie
# date: Feb, 2020
import os
from models.training import BaseTrainer
import torch
from net_utils.libs import get_rotation_matrix_gt, get_mask_status
from configs.data_config import NYU37_TO_PIX3D_CLS_MAPPING

class Trainer(BaseTrainer):
    '''
    Trainer object for total3d.
    '''
    def eval_step(self, data):
        '''
        performs a step in evaluation
        :param data (dict): data dictionary
        :return:
        '''
        loss = self.compute_loss(data)
        loss['total'] = loss['total'].item()
        return loss

    def visualize_step(self, epoch, phase, iter, data):
        ''' Performs a visualization step.
        '''
        pass

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

            obj_masks = data['boxes_batch']['mask']

            mask_status = get_mask_status(obj_masks, split)

            mask_flag = 1
            if 1 not in mask_status:
                mask_flag = 0

            # Notice: we should conclude the NYU37 classes into pix3d (9) classes before feeding into the network.
            cls_codes = torch.zeros([size_cls.size(0), 9]).to(device)
            cls_codes[range(size_cls.size(0)), [NYU37_TO_PIX3D_CLS_MAPPING[cls.item()] for cls in
                                                torch.argmax(size_cls, dim=1)]] = 1

            patch_for_mesh = patch[mask_status.nonzero()]
            cls_codes_for_mesh = cls_codes[mask_status.nonzero()]

            '''calculate loss from the interelationship between object and layout.'''
            bdb2D_from_3D_gt = data['boxes_batch']['bdb2D_from_3D'].float().to(device)
            bdb2D_pos = data['boxes_batch']['bdb2D_pos'].float().to(device)

            joint_input = {'bdb3D':bdb3D, 'K':K, 'depth_maps':depth_maps, 'cam_R_gt':cam_R_gt,
                           'obj_masks':obj_masks, 'mask_status':mask_status, 'mask_flag':mask_flag, 'patch_for_mesh':patch_for_mesh,
                           'cls_codes_for_mesh':cls_codes_for_mesh, 'bdb2D_from_3D_gt':bdb2D_from_3D_gt, 'bdb2D_pos':bdb2D_pos}

        '''output data'''
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'layout_estimation':
            return layout_input
        elif self.cfg.config[self.cfg.config['mode']]['phase'] == 'object_detection':
            return object_input
        elif self.cfg.config[self.cfg.config['mode']]['phase'] == 'joint':
            return {**layout_input, **object_input, **joint_input}
        else:
            raise NotImplementedError

    def compute_loss(self, data):
        '''
        compute the overall loss.
        :param data (dict): data dictionary
        :return:
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        '''computer losses'''
        loss = self.net.loss(est_data, data)
        return loss
