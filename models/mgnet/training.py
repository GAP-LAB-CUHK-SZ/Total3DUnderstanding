# Trainer for Mesh Generation Net.
# author: ynie
# date: Feb, 2020
from models.training import BaseTrainer


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
        img = data['img'].to(device)
        cls = data['cls'].float().to(device)
        mesh_points = data['mesh_points'].float().to(device)
        densities = data['densities'].float().to(device)
        sequence_id = data['sequence_id']
        return {'img':img, 'cls':cls, 'mesh_points':mesh_points, 'densities':densities, 'sequence_id':sequence_id}

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
