# Base trainer for methods.
# author: ynie
# date: Feb, 2020

class BaseTrainer(object):
    '''
    Base trainer for all networks.
    '''
    def __init__(self, cfg, net, optimizer, device=None):
        self.cfg = cfg
        self.net = net
        self.optimizer = optimizer
        self.device = device

    def show_lr(self):
        '''
        display current learning rates
        :return:
        '''
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        self.cfg.log_string('Current learning rates are: ' + str(lrs) + '.')

    def train_step(self, data):
        '''
        performs a step training
        :param data (dict): data dictionary
        :return:
        '''
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss['total'].backward()
        self.optimizer.step()

        loss['total'] = loss['total'].item()
        return loss

    def eval_loss_parser(self, loss_recorder):
        '''
        get the eval
        :param loss_recorder: loss recorder for all losses.
        :return:
        '''
        return loss_recorder['total'].avg

    def compute_loss(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize_step(self, *args, **kwargs):
        ''' Performs a visualization step.
        '''
        raise NotImplementedError