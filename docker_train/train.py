
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
import torch.cuda.amp as amp

from models import build_model
from imagenet.imagenet_cv2 import ImageNet
#  from imagenet.imagenet import ImageNet
from eval import eval_model
from meters import TimeMeter, AvgMeter
from logger import setup_logger
from ops import EMA, MixUper, CutMixer
from pytorch_loss import LabelSmoothSoftmaxCEV3, OnehotEncoder
from rmsprop_tf import RMSpropTF
from lr_scheduler import (
        WarmupExpLrScheduler, WarmupStepLrScheduler, WarmupCosineLrScheduler)
from cross_entropy import (
        SoftmaxCrossEntropyV2,
        SoftmaxCrossEntropyV1
    )


#  from config.spinenet49 import *
#  from config.spinenet49s import *
#  from config.pa_resnet50 import *
#  from config.pa_resnet101 import *
#  from config.resnet50 import *
#  from config.resnet101 import *
#  from config.frelu_resnet50 import *
#  from config.frelu_resnet101 import *
#  from config.xception41 import *
#  from config.xception65 import *
from config.xception71 import *
#  from config.wa_resnet50 import *
#  from config.askc_resnet101 import *
#  from config.se_resnet50 import *
#  from config.se_pa_resnet50 import *
#  from config.se_pa_resnet101 import *

#  from config.effnetb0 import *
#  from config.effnetb0_lite import *
#  from config.effnetb1 import *
#  from config.effnetb2 import *
#  from config.effnetb2_lite import *
#  from config.effnetb4 import *
#  from config.effnetb6 import *
#  from config.ushape_effnetb0 import *


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


def set_optimizer(model, opt_type, opt_args, schdlr_type, schdlr_args):
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
    opt_dict = {'SGD': torch.optim.SGD, 'RMSpropTF': RMSpropTF}
    schdlr_dict = {'ExpLr': WarmupExpLrScheduler,
                'StepLr': WarmupStepLrScheduler,
                'CosineLr': WarmupCosineLrScheduler,
                }

    optim = opt_dict[opt_type](params_list, **opt_args)
    scheduler = schdlr_dict[schdlr_type](optim, **schdlr_args)
    ## scheduler

    return optim, scheduler


def main():
    global n_eval_epoch

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
        sampler_val, batchsize * 1, drop_last=False
    )
    dl_eval = DataLoader(
        dataset_eval, batch_sampler=batch_sampler_val,
        num_workers=4, pin_memory=True
    )
    n_iters_per_epoch = len(dataset_train) // n_gpus // batchsize
    n_iters = n_epoches * n_iters_per_epoch


    ## model
    model = build_model(**model_args)
    model.cuda()

    ## sync bn
    if use_sync_bn: model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #  crit = nn.CrossEntropyLoss()
    #  crit = LabelSmoothSoftmaxCEV3(lb_smooth)
    crit = SoftmaxCrossEntropyV1()

    ## optimizer
    optim, scheduler = set_optimizer(model,
            opt_type, opt_args, schdlr_type, schdlr_args)
    scheduler.update_by_iter(n_iters_per_epoch)

    ## mixed precision
    scaler = amp.GradScaler()

    ## ema
    ema = EMA(model, ema_alpha)

    ## ddp training
    local_rank = dist.get_rank()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank, ], output_device=local_rank
    )

    ## log meters
    time_meter = TimeMeter(n_iters)
    loss_meter = AvgMeter()
    logger = logging.getLogger()

    # for mixup
    label_encoder = OnehotEncoder(n_classes=model_args['n_classes'], lb_smooth=lb_smooth)
    mixuper = MixUper(mixup_alpha)
    cutmixer = CutMixer(cutmix_beta)

    ## train loop
    for e in range(n_epoches):
        sampler_train.set_epoch(e)
        model.train()
        for idx, (im, lb) in enumerate(dl_train):
            im, lb= im.cuda(non_blocking=True), lb.cuda(non_blocking=True)

            lb = label_encoder(lb)
            #  im, lb = mixuper(im, lb)
            #  im, lb = cutmixer(im, lb)

            optim.zero_grad()
            with amp.autocast(enabled=use_mixed_precision):
                logits = model(im)
                loss = crit(logits, lb) #+ cal_l2_loss(model, weight_decay)
            scaler.scale(loss).backward()

            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optim)
            scaler.update()
            torch.cuda.synchronize()
            ema.update_params()
            time_meter.update()
            loss_meter.update(loss.item())
            if (idx + 1) % 200 == 0:
                t_intv, eta = time_meter.get()
                lr_log = scheduler.get_lr()
                lr_log = sum(lr_log) / len(lr_log)
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
        #  torch.save(model.module.state_dict(), './res/model_final_naive.pth')
        #  torch.save(ema.ema_model.state_dict(), './res/model_final_ema.pth')
        torch.save(model.module.get_states(), './res/model_final_naive.pth')
        torch.save(ema.ema_model.get_states(), './res/model_final_ema.pth')


def evaluate(ema, dl_eval):
    model = ema.ema_model
    acc_1_ema, acc_5_ema = eval_model(model, dl_eval)
    model = ema.model
    acc_1, acc_5 = eval_model(model, dl_eval)
    torch.cuda.empty_cache()
    return acc_1, acc_5, acc_1_ema, acc_5_ema



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
    setup_logger(model_args['model_type'], './res/')
    main()
    dist.barrier()
