# Testing functions.
# author: ynie
# date: April, 2020
from net_utils.utils import LossRecorder
from time import time
import numpy as np
import torch

def test_func(cfg, tester, test_loader):
    '''
    test function.
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    batch_size = cfg.config[cfg.config['mode']]['batch_size']
    loss_recorder = LossRecorder(batch_size)
    cfg.log_string('-'*100)
    for iter, data in enumerate(test_loader):
        loss = tester.test_step(data)

        # visualize intermediate results.
        tester.visualize_step(0, cfg.config['mode'], iter, data)

        loss_recorder.update_loss(loss)

        if ((iter + 1) % cfg.config['log']['print_step']) == 0:
            cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (
            cfg.config['mode'], 0, iter + 1, len(test_loader), str(loss)))

    return loss_recorder.loss_recorder

def test(cfg, tester, test_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param tester: specific tester for networks
    :param test_loader: dataloader for testing
    :return:
    '''
    cfg.log_string('-' * 100)
    # set mode
    tester.net.train(cfg.config['mode'] == 'train')
    start = time()
    with torch.no_grad():
        test_loss_recoder = test_func(cfg, tester, test_loader)
    cfg.log_string('Test time elapsed: (%f).' % (time()-start))
    for key, test_loss in test_loss_recoder.items():
        cfg.log_string('Test loss (%s): %f' % (key, test_loss.avg))
