
import os
import os.path as osp
import pickle
import argparse
import logging
import cv2
import numpy as np
import random
import math

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from apex import amp, parallel

from efficientnet_refactor import EfficientNet
from resnet import ResNet50
from imagenet.imagenet_cv2 import ImageNet
#  from imagenet.imagenet import ImageNet
from eval import evaluate
from meters import TimeMeter, AvgMeter
from logger import setup_logger
from ema import EMA
from label_smooth import LabelSmoothSoftmaxCEV2
from rmsprop_tf import RMSpropTF
from lr_scheduler import WarmupExpLrScheduler, WarmupStepLrScheduler


### bs=32, lr0, 8/23
### bs=128, lr0 x 4, 单卡, 6/19
### bs=128, lr0 x 4, 多卡, 9/25
### bs=128, lr0 x 4, 多卡, sampler, 9/25
### bs=128, lr0 x 4, 单卡, sampler, 5/17
### bs=128, lr0 x 4, 多卡, sampler, 2/9
### bs=128, lr0 x 4 x 4, 多卡, sampler, 6/19

### 所以多卡的时候, 学习率还是跟单卡一样调整, 整体batchsize变大之后, lr也要放大


random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system') # this would make it stuck when program is done


def init_model_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            #  nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if not module.bias is None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            if hasattr(module, 'last_bn') and module.last_bn:
                nn.init.zeros_(module.weight)
            else:
                nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)  # fan-out
            fan_in = 0
            init_range = 1.0 / math.sqrt(fan_in + fan_out)
            #  module.weight.data.uniform_(-init_range, init_range)
            module.weight.data.normal_(mean=0.0, std=0.01)
            #  nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)


def cal_l2_loss(model, weight_decay):
    l2loss = 0
    wd_params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            wd_params.append(module.weight)
            if not module.bias is None: wd_params.append(module.bias)
    for param in wd_params:
        l2loss += weight_decay * (param ** 2).sum()
    return 0.5 * l2loss


def set_optimizer(model, lr, wd, momentum, nesterov):
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        param_len = param.dim()
        if param_len == 4 or param_len == 2:
            wd_params.append(param)
        elif param_len == 1:
            non_wd_params.append(param)
        else:
            print(name)
    params_list = [
        {'params': wd_params},
        {'params': non_wd_params, 'weight_decay': 0},
    ]
    optim = RMSpropTF(
        params_list,
        lr=lr,
        alpha=0.9,
        eps=1e-3,
        weight_decay=wd,
        momentum=momentum
    )
    #  optim = torch.optim.SGD(
    #      params_list, lr=lr, weight_decay=wd, momentum=momentum, nesterov=nesterov
    #  )
    return optim


def main():
    n_gpus = torch.cuda.device_count()
    batchsize = 128
    n_epoches = 350
    n_eval_epoch = 1
    lr = 1.6e-2 * (batchsize / 256) * n_gpus
    weight_decay = 1e-5
    opt_wd = 1e-5
    momentum = 0.9
    warmup = 'linear'
    warmup_ratio = 0
    datapth = './imagenet/'
    model_type = 'b0'
    n_classes = 1000
    cropsize = 224
    num_workers = 4
    ema_alpha = 0.9999
    fp16_level = 'O1'
    use_sync_bn = False
    nesterov = True

    ## resnet50 params
    #  n_gpus = torch.cuda.device_count()
    #  batchsize = 128
    #  n_epoches = 100
    #  n_eval_epoch = 1
    #  lr = 0.1 * (batchsize / 128) * n_gpus
    #  opt_wd = 1e-4
    #  nesterov = True
    #  momentum = 0.9
    #  warmup = 'linear'
    #  warmup_ratio = 0.1
    #  datapth = './imagenet/'
    #  n_classes = 1000
    #  cropsize = 224
    #  num_workers = 4
    #  ema_alpha = 0.9999
    #  fp16_level = 'O1'
    #  use_sync_bn = False

    ## dataloader
    dataset_train = ImageNet(datapth, mode='train', cropsize=cropsize)
    sampler_train = torch.utils.data.distributed.DistributedSampler(
        dataset_train, shuffle=True)
    batch_sampler_train = torch.utils.data.sampler.BatchSampler(
        sampler_train, batchsize, drop_last=True
    )
    dl_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train, num_workers=num_workers, pin_memory=True
    )
    dataset_eval = ImageNet(datapth, mode='val', cropsize=cropsize)
    sampler_val = torch.utils.data.distributed.DistributedSampler(
        dataset_eval, shuffle=False)
    batch_sampler_val = torch.utils.data.sampler.BatchSampler(
        sampler_val, batchsize * 2, drop_last=False
    )
    dl_eval = DataLoader(
        dataset_eval, batch_sampler=batch_sampler_val,
        num_workers=4, pin_memory=True
    )
    n_iters_per_epoch = len(dataset_train) // n_gpus // batchsize
    n_iters = n_epoches * n_iters_per_epoch


    ## model
    model = EfficientNet(model_type, n_classes)
    #  model = ResNet50()

    #  from models import create_model
    #  model = create_model('efficientnet_b0',
    #          drop_connect_rate=0.2,
    #          global_pool='avg',
    #          drop_rate=0.2)

    ## sync bn
    #  if use_sync_bn: model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    init_model_weights(model)
    model.cuda()
    if use_sync_bn: model = parallel.convert_syncbn_model(model)
    #  crit = nn.CrossEntropyLoss()
    crit = LabelSmoothSoftmaxCEV2()

    ## optimizer
    optim = set_optimizer(model, lr, opt_wd, momentum, nesterov=nesterov)

    ## apex
    model, optim = amp.initialize(model, optim, opt_level=fp16_level)

    ## ema
    ema = EMA(model, ema_alpha)

    ## ddp training
    model = parallel.DistributedDataParallel(model, delay_allreduce=True)
    #  local_rank = dist.get_rank()
    #  model = nn.parallel.DistributedDataParallel(
    #      model, device_ids=[local_rank, ], output_device=local_rank
    #  )

    ## log meters
    time_meter = TimeMeter(n_iters)
    loss_meter = AvgMeter()
    logger = logging.getLogger()

    ## scheduler
    scheduler = WarmupExpLrScheduler(
        optim, gamma=0.97, interval=n_iters_per_epoch * 2.4,
        warmup_iter=n_iters_per_epoch * 5, warmup=warmup,
        warmup_ratio=warmup_ratio
    )
    #  scheduler = WarmupStepLrScheduler(
    #      optim, milestones=[n_iters_per_epoch * el for el in [30, 60, 90]],
    #      warmup_iter=n_iters_per_epoch * 0, warmup=warmup,
    #      warmup_ratio=warmup_ratio
    #  )

    ## train loop
    for e in range(n_epoches):
        sampler_train.set_epoch(e)
        model.train()
        for idx, (im, lb) in enumerate(dl_train):
            im, lb= im.cuda(), lb.cuda()
            optim.zero_grad()
            logits = model(im)
            loss = crit(logits, lb) #+ cal_l2_loss(model, weight_decay)
            #  loss.backward()
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
            optim.step()
            torch.cuda.synchronize()
            ema.update_params()
            time_meter.update()
            loss_meter.update(loss.item())
            if (idx + 1) % 200 == 0:
                t_intv, eta = time_meter.get()
                lr_log = scheduler.get_lr_ratio() * lr
                msg = 'epoch: {}, iter: {}, lr: {:.4f}, loss: {:.4f}, time: {:.2f}, eta: {}'.format(
                    e + 1, idx + 1, lr_log, loss_meter.get()[0], t_intv, eta)
                logger.info(msg)
            scheduler.step()
        torch.cuda.empty_cache()
        if (e + 1) % n_eval_epoch == 0:
            if e > 50: n_eval_epoch = 5
            logger.info('evaluating...')
            acc_1, acc_5, acc_1_ema, acc_5_ema = evaluate(ema, dl_eval)
            msg = 'epoch: {}, naive_acc1: {:.4}, naive_acc5: {:.4}, ema_acc1: {:.4}, ema_acc5: {:.4}'.format(e + 1, acc_1, acc_5, acc_1_ema, acc_5_ema)
            logger.info(msg)
    if dist.is_initialized() and dist.get_rank() == 0:
        torch.save(model.module.state_dict(), './res/model_final.pth')
        torch.save(ema.ema_model.state_dict(), './res/model_final_ema.pth')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank',
                       dest='local_rank',
                       type=int,
                       default=-1,)
    return parse.parse_args()


def init_dist(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )


if __name__ == '__main__':
    args = parse_args()
    init_dist(args)
    setup_logger('./res/')
    main()
    dist.barrier()
