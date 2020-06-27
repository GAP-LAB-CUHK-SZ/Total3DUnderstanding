# Tester for Mesh Generation Net.
# author: ynie
# date: April, 2020
import os
from models.testing import BaseTester
import torch
from .training import Trainer
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
from libs.tools import write_obj
dist_chamfer = ChamferDistance()


class Tester(BaseTester, Trainer):
    '''
    Tester object for SCNet.
    '''
    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)

    def get_metric_values(self, est_data, data):
        chamfer_values = []

        final_mesh = est_data['mesh_coordinates_results'][-1].transpose(1, 2)
        for index in range(final_mesh.shape[0]):
            current_faces = est_data['faces'][index]
            current_coordinates = final_mesh[index]

            dist1, dist2 = dist_chamfer(data['mesh_points'][index].unsqueeze(0), current_coordinates.unsqueeze(0))[:2]
            cls_id = data['cls'][index].nonzero()[0][0].item()
            chamfer_values.append(((torch.mean(dist1)) + (torch.mean(dist2))).item())
            if self.cfg.config['log']['save_results']:
                file_path = os.path.join(self.cfg.config['log']['vis_path'], '%s_%s.obj' % (data['sequence_id'][index].item(), cls_id))

                mesh_obj = {'v': current_coordinates.cpu().numpy(),
                            'f': current_faces.cpu().numpy()}

                write_obj(file_path, mesh_obj)
        return {'Avg_Chamfer': chamfer_values}

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