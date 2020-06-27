# Training functions.
# author: ynie
# date: Feb, 2020

from net_utils.utils import LossRecorder
from time import time

def train_epoch(cfg, epoch, trainer, dataloaders):
    '''
    train by epoch
    :param cfg: configuration file
    :param epoch: epoch id.
    :param trainer: specific trainer for networks
    :param dataloaders: dataloader for training and validation
    :return:
    '''
    for phase in ['train', 'test']:
        dataloader = dataloaders[phase]
        batch_size = cfg.config[phase]['batch_size']
        loss_recorder = LossRecorder(batch_size)
        # set mode
        trainer.net.train(phase == 'train')
        # set subnet mode
        trainer.net.set_mode()
        cfg.log_string('-' * 100)
        cfg.log_string('Switch Phase to %s.' % (phase))
        cfg.log_string('-'*100)
        for iter, data in enumerate(dataloader):
            if phase == 'train':
                loss = trainer.train_step(data)
            else:
                loss = trainer.eval_step(data)

            # visualize intermediate results.
            if ((iter + 1) % cfg.config['log']['vis_step']) == 0:
                trainer.visualize_step(epoch, phase, iter, data)

            loss_recorder.update_loss(loss)

            if ((iter + 1) % cfg.config['log']['print_step']) == 0:
                cfg.log_string('Process: Phase: %s. Epoch %d: %d/%d. Current loss: %s.' % (phase, epoch, iter + 1, len(dataloader), str(loss)))

        cfg.log_string('=' * 100)
        for loss_name, loss_value in loss_recorder.loss_recorder.items():
            cfg.log_string('Currently the last %s loss (%s) is: %f' % (phase, loss_name, loss_value.avg))
        cfg.log_string('=' * 100)

    return loss_recorder.loss_recorder

def train(cfg, trainer, scheduler, checkpoint, train_loader, test_loader):
    '''
    train epochs for network
    :param cfg: configuration file
    :param scheduler: scheduler for optimizer
    :param trainer: specific trainer for networks
    :param checkpoint: network weights.
    :param train_loader: dataloader for training
    :param test_loader: dataloader for testing
    :return:
    '''
    start_epoch = scheduler.last_epoch
    total_epochs = cfg.config['train']['epochs']
    min_eval_loss = checkpoint.get('min_loss')

    dataloaders = {'train': train_loader, 'test': test_loader}

    for epoch in range(start_epoch, total_epochs):
        cfg.log_string('-' * 100)
        cfg.log_string('Epoch (%d/%s):' % (epoch + 1, total_epochs))
        trainer.show_lr()
        start = time()
        eval_loss_recorder = train_epoch(cfg, epoch + 1, trainer, dataloaders)
        eval_loss = trainer.eval_loss_parser(eval_loss_recorder)
        scheduler.step(eval_loss)
        cfg.log_string('Epoch (%d/%s) Time elapsed: (%f).' % (epoch + 1, total_epochs, time()-start))

        # save checkpoint
        checkpoint.register_modules(epoch=epoch, min_loss=eval_loss)
        checkpoint.save('last')
        cfg.log_string('Saved the latest checkpoint.')
        if epoch==-1 or eval_loss<min_eval_loss:
            checkpoint.save('best')
            min_eval_loss = eval_loss
            cfg.log_string('Saved the best checkpoint.')
            cfg.log_string('=' * 100)
            for loss_name, loss_value in eval_loss_recorder.items():
                cfg.log_string('Currently the best test loss (%s) is: %f' % (loss_name, loss_value.avg))
            cfg.log_string('=' * 100)