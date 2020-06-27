# Total3D: model loader
# author: ynie
# date: Feb, 2020

from models.registers import METHODS, MODULES, LOSSES
from models.network import BaseNetwork
import torch
from torch import nn
from configs.data_config import obj_cam_ratio

@METHODS.register_module
class TOTAL3D(BaseNetwork):

    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        phase_names = []
        if cfg.config[cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            phase_names += ['layout_estimation']
        if cfg.config[cfg.config['mode']]['phase'] in ['object_detection', 'joint']:
            phase_names += ['object_detection']
        if cfg.config[cfg.config['mode']]['phase'] in ['joint']:
            phase_names += ['mesh_reconstruction']

        if (not cfg.config['model']) or (not phase_names):
            cfg.log_string('No submodule found. Please check the phase name and model definition.')
            raise ModuleNotFoundError('No submodule found. Please check the phase name and model definition.')

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            if phase_name not in phase_names:
                continue
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1)))

        '''Add joint loss'''
        setattr(self, 'joint_loss', LOSSES.get('JointLoss', 'Null')(1))

        '''Multi-GPU setting'''
        # Note that for object_detection, we should extract relational features, thus it does not support parallel training.
        if cfg.config[cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            self.layout_estimation = nn.DataParallel(self.layout_estimation)
        if cfg.config[cfg.config['mode']]['phase'] in ['joint']:
            self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def forward(self, data):

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            pitch_reg_result, roll_reg_result, \
            pitch_cls_result, roll_cls_result, \
            lo_ori_reg_result, lo_ori_cls_result, \
            lo_centroid_result, lo_coeffs_result = self.layout_estimation(data['image'])

            layout_output = {'pitch_reg_result':pitch_reg_result, 'roll_reg_result':roll_reg_result,
                             'pitch_cls_result':pitch_cls_result, 'roll_cls_result':roll_cls_result,
                             'lo_ori_reg_result':lo_ori_reg_result, 'lo_ori_cls_result':lo_ori_cls_result,
                             'lo_centroid_result':lo_centroid_result, 'lo_coeffs_result':lo_coeffs_result}

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['object_detection', 'joint']:
            size_reg_result, \
            ori_reg_result, ori_cls_result, \
            centroid_reg_result, centroid_cls_result, \
            offset_2D_result = self.object_detection(data['patch'], data['size_cls'], data['g_features'],
                                                     data['split'], data['rel_pair_counts'])
            object_output = {'size_reg_result':size_reg_result, 'ori_reg_result':ori_reg_result,
                             'ori_cls_result':ori_cls_result, 'centroid_reg_result':centroid_reg_result,
                             'centroid_cls_result':centroid_cls_result, 'offset_2D_result':offset_2D_result}

        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['joint']:
            # predict meshes
            if self.cfg.config['mode']=='train':
                if data['mask_flag'] == 1:
                    mesh_output = self.mesh_reconstruction(data['patch_for_mesh'], data['cls_codes_for_mesh'])[0][-1]
                    # convert to SUNRGBD coordinates
                    mesh_output[:, 2, :] *= -1
                else:
                    mesh_output = None
                mesh_output = {'meshes': mesh_output}
            else:
                mesh_output, _, _, _, _, out_faces = self.mesh_reconstruction(data['patch'], data['cls_codes'])
                mesh_output = mesh_output[-1]
                # convert to SUNRGBD coordinates
                mesh_output[:, 2, :] *= -1
                mesh_output = {'meshes': mesh_output, 'out_faces': out_faces}

        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'layout_estimation':
            return layout_output
        elif self.cfg.config[self.cfg.config['mode']]['phase'] == 'object_detection':
            return object_output
        elif self.cfg.config[self.cfg.config['mode']]['phase'] == 'joint':
            return {**layout_output, **object_output, **mesh_output}
        else:
            raise NotImplementedError

    def loss(self, est_data, gt_data):
        '''
        calculate loss of est_out given gt_out.
        '''
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['layout_estimation', 'joint']:
            layout_loss, layout_results = self.layout_estimation_loss(est_data, gt_data, self.cfg.bins_tensor)

            total_layout_loss = sum(layout_loss.values())
            for key, value in layout_loss.items():
                layout_loss[key] = value.item()
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['object_detection', 'joint']:
            object_loss = self.object_detection_loss(est_data, gt_data)

            total_object_loss = sum(object_loss.values())
            for key, value in object_loss.items():
                object_loss[key] = value.item()
        if self.cfg.config[self.cfg.config['mode']]['phase'] in ['joint']:
            joint_loss, extra_results = self.joint_loss(est_data, gt_data, self.cfg.bins_tensor, layout_results)
            mesh_loss = self.mesh_reconstruction_loss(est_data, gt_data, extra_results)

            total_joint_loss = sum(joint_loss.values()) + sum(mesh_loss.values())
            for key, value in mesh_loss.items():
                mesh_loss[key] = float(value)
            for key, value in joint_loss.items():
                joint_loss[key] = value.item()

        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'layout_estimation':
            return {'total':total_layout_loss, **layout_loss}
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'object_detection':
            return {'total':total_object_loss, **object_loss}
        if self.cfg.config[self.cfg.config['mode']]['phase'] == 'joint':
            total3d_loss = total_object_loss + total_joint_loss + obj_cam_ratio * total_layout_loss
            return {'total':total3d_loss, **layout_loss, **object_loss, **mesh_loss, **joint_loss}
        else:
            raise NotImplementedError