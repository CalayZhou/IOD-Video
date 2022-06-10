from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import torch
import torch.utils.data
from opts import opts
from utils.model import create_model, load_model, save_model, load_imagenet_pretrained_model
from trainer.logger import Logger
from datasets.init_dataset import get_dataset
from trainer.trainer import Trainer
import numpy as np
import random
import tensorboardX


GLOBAL_SEED = 317

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)

def main(opt):
    set_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    print('dataset: ' + opt.dataset + '   task:  ' + opt.task)
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset(opt, Dataset)

    #log
    train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train'))
    epoch_train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train_epoch'))
    val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val'))
    epoch_val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val_epoch'))

    logger = Logger(opt, epoch_train_writer, epoch_val_writer)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    #model define
    model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.K)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = opt.start_epoch

    #load from the imagenet pre-trained model
    if opt.pretrain_model == 'imagenet':
        model = load_imagenet_pretrained_model(opt, model)
    else:
        print("there is no pretrained model, init from random parameter.")

    #load from the already trained model
    if opt.load_model != '':
        model, optimizer, _, _ = load_model(model, opt.load_model, optimizer, opt.lr)

    #Trainer Class
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('training...')
    print('GPU allocate:', opt.chunk_sizes)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print('epoch is ', epoch)

        #Dataset Class
        Dataset = get_dataset(opt.dataset)
        opt = opts().update_dataset(opt, Dataset)

        #DataLoader
        train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
        ) 
        val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
        )  

        # train
        log_dict_train = trainer.train(epoch, train_loader, train_writer)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'train')
            logger.write('train: {} {:8f} | '.format(k, v))
        logger.write('\n')

        # save the model
        if opt.save_all:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            model_name = 'model_[{}]_{}.pth'.format(epoch, time_str)
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])
        else:
            model_name = 'model_last.pth'
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])

        # evaluate the model
        if opt.val_epoch:
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader, val_writer)
            for k, v in log_dict_val.items():
                logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'val')
                logger.write('val: {} {:8f} | '.format(k, v))
        logger.write('\n')

        #decrese the learning rate
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            logger.write('Drop LR to ' + str(lr) + '\n')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    logger.close()


if __name__ == '__main__':
    os.system("rm -rf tmp")
    opt = opts().parse()
    main(opt)
