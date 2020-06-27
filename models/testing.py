# Base tester for methods.
# author: ynie
# date: April, 2020

class BaseTester(object):
    '''
    Base tester for all networks.
    '''
    def __init__(self, cfg, net, device=None):
        self.cfg = cfg
        self.net = net
        self.device = device

    def visualize_step(self, *args, **kwargs):
        ''' Performs a visualization step.
        '''
        raise NotImplementedError

    def get_metric_values(self, est_data, gt_data):
        ''' Performs a evaluation step.
        '''
        # camera orientation error
        raise NotImplementedError




