# Configure trainer and tester
# author: ynie
# date: Feb, 2020
from .training import Trainer
from .testing import Tester
from .dataloader import Total3D_dataloader

def get_trainer(cfg, net, optimizer, device=None):
    return Trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)

def get_tester(cfg, net, device=None):
    return Tester(cfg=cfg, net=net, device=device)

def get_dataloader(config, mode):
    return Total3D_dataloader(config=config, mode=mode)