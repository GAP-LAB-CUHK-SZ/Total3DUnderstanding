# Optimizer definitions
# author: ynie
# date: Feb, 2020
import torch

def has_optim_in_children(subnet):
    '''
    check if there is specific optim parameters in a subnet.
    :param subnet:
    :return:
    '''
    label = False
    for module in subnet.children():
        if hasattr(module, 'optim_spec') and module.optim_spec:
            label = True
            break
        else:
            label = has_optim_in_children(module)

    return label

def find_optim_module(net):
    '''
    classify modules in a net into has specific optim specs or not.
    :param net:
    :return:
    '''
    module_optim_pairs = []
    other_modules = []
    for module in net.children():
        if hasattr(module, 'optim_spec'):
            module_optim_pairs += [{'module':module, 'optim_spec':module.optim_spec}]
        elif not has_optim_in_children(module):
            other_modules += [module]
        else:
            module_optim_pairs += find_optim_module(module)[0]
            other_modules += find_optim_module(module)[1]

    return module_optim_pairs, other_modules

def load_scheduler(config, optimizer):
    '''
    get scheduler for optimizer.
    :param config: configuration file
    :param optimizer: torch optimizer
    :return:
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=float(config['scheduler']['factor']),
                                                           patience=config['scheduler']['patience'],
                                                           threshold=float(config['scheduler']['threshold']),
                                                           verbose=True)
    return scheduler

def load_optimizer(config, net):
    '''
    get optimizer for networks
    :param config: configuration file
    :param model: nn.Module network
    :return:
    '''

    module_optim_pairs, other_modules = find_optim_module(net)
    default_optim_spec = config['optimizer']

    optim_params = []

    if config['optimizer']['method'] == 'Adam':
        '''collect parameters with specific optimizer spec'''
        for module in module_optim_pairs:
            optim_params.append({'params': filter(lambda p: p.requires_grad, module['module'].parameters()),
                                 'lr': float(module['optim_spec']['lr']),
                                 'betas': tuple(module['optim_spec']['betas']),
                                 'eps': float(module['optim_spec']['eps']),
                                 'weight_decay': float(module['optim_spec']['weight_decay'])})

        '''collect parameters with default optimizer spec'''
        other_params = list()
        for module in other_modules:
            other_params += list(module.parameters())

        optim_params.append({'params': filter(lambda p: p.requires_grad, other_params)})

        '''define optimizer'''
        optimizer = torch.optim.Adam(optim_params,
                                     lr=float(default_optim_spec['lr']),
                                     betas=tuple(default_optim_spec['betas']),
                                     eps=float(default_optim_spec['eps']),
                                     weight_decay=float(default_optim_spec['weight_decay']))

    else:
        # use SGD optimizer.
        for module in module_optim_pairs:
            optim_params.append({'params': filter(lambda p: p.requires_grad, module['module'].parameters()),
                                 'lr': float(module['optim_spec']['lr'])})

        other_params = list()
        for module in other_modules:
            other_params += list(module.parameters())

        optim_params.append({'params': filter(lambda p: p.requires_grad, other_params)})
        optimizer = torch.optim.SGD(optim_params,
                                    lr=config['optimizer']['lr'],
                                    momentum=0.9)

    return optimizer
