# Testing script
# author: ynie
# date: April, 2020
from net_utils.utils import load_device, load_model, load_tester, load_dataloader
from net_utils.utils import CheckpointIO
from test_epoch import test
from configs.config_utils import mount_external_config

def run(cfg):
    '''Begin to run network.'''
    checkpoint = CheckpointIO(cfg)

    '''Mount external config data'''
    cfg = mount_external_config(cfg)

    '''Load save path'''
    cfg.log_string('Data save path: %s' % (cfg.save_path))

    '''Load device'''
    cfg.log_string('Loading device settings.')
    device = load_device(cfg)

    '''Load data'''
    cfg.log_string('Loading dataset.')
    test_loader = load_dataloader(cfg.config, mode='test')

    '''Load net'''
    cfg.log_string('Loading model.')
    net = load_model(cfg, device=device)
    checkpoint.register_modules(net=net)
    cfg.log_string(net)

    '''Load existing checkpoint'''
    checkpoint.parse_checkpoint()

    '''Load tester'''
    cfg.log_string('Loading tester.')
    tester = load_tester(cfg=cfg, net=net, device=device)

    '''Start to test'''
    cfg.log_string('Start to test.')
    cfg.log_string('Total number of parameters in {0:s}: {1:d}.'.format(cfg.config['method'], sum(p.numel() for p in net.parameters())))

    test(cfg=cfg, tester=tester, test_loader=test_loader)

    cfg.write_config()
    cfg.log_string('Testing finished.')