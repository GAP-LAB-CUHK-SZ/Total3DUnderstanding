# Base Network for other networks.
# author: ynie
# date: Feb, 2020

from models.registers import MODULES, LOSSES
import torch.nn as nn

class BaseNetwork(nn.Module):
    '''
    Base Network Module for other networks
    '''
    def __init__(self, cfg):
        '''
        load submodules for the network.
        :param config: customized configurations.
        '''
        super(BaseNetwork, self).__init__()
        self.cfg = cfg

        '''load network blocks'''
        for phase_name, net_spec in cfg.config['model'].items():
            method_name = net_spec['method']
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            subnet = MODULES.get(method_name)(cfg.config, optim_spec)
            self.add_module(phase_name, subnet)

            '''load corresponding loss functions'''
            setattr(self, phase_name + '_loss', LOSSES.get(self.cfg.config['model'][phase_name]['loss'], 'Null')(
                self.cfg.config['model'][phase_name].get('weight', 1)))

        '''freeze submodules or not'''
        self.freeze_modules(cfg)

    def freeze_modules(self, cfg):
        '''
        Freeze modules in training
        '''
        if cfg.config['mode'] == 'train':
            freeze_layers = cfg.config['train']['freeze']
            for layer in freeze_layers:
                if not hasattr(self, layer):
                    continue
                for param in getattr(self, layer).parameters():
                    param.requires_grad = False
                cfg.log_string('The module: %s is fixed.' % (layer))

    def set_mode(self):
        '''
        Set train/eval mode for the network.
        :param phase: train or eval
        :return:
        '''
        freeze_layers = self.cfg.config['train']['freeze']
        for name, child in self.named_children():
            if name in freeze_layers:
                child.train(False)

        # turn off BatchNorm if batch_size == 1.
        if self.cfg.config[self.cfg.config['mode']]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()

    def load_weight(self, pretrained_model):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model.items() if
                           k in model_dict}
        self.cfg.log_string(
            str(set([key.split('.')[0] for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_optim_spec(self, config, net_spec):
        # load specific optimizer parameters
        if config['mode'] == 'train':
            if 'optimizer' in net_spec.keys():
                optim_spec = net_spec['optimizer']
            else:
                optim_spec = config['optimizer']  # else load default optimizer
        else:
            optim_spec = None

        return optim_spec

    def forward(self, *args, **kwargs):
        ''' Performs a forward step.
        '''
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        ''' calculate losses.
        '''
        raise NotImplementedError