# Register different methods, and relevant modules
# author: ynie
# date: Feb, 2020

from net_utils.registry import Registry

METHODS = Registry('method')
MODULES = Registry('module')
LOSSES = Registry('loss')