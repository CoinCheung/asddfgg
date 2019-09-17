import os
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist
from apex import amp

from efficientnet import EfficientNet
from imagenet.imagenet import ImageNet
from lr_scheduler import WarmupExpLrScheduler
from meters import TimeMeter, AvgMeter
from ema import EMA
from checkpoint import save_ckpt, load_ckpt


model_type = 'b0'
cropsize = 224
n_gpus = int(os.environ['num_of_gpus'])
bs_per_gpu = 256
batchsize = bs_per_gpu * n_gpus
n_train_imgs = 1281167
n_epoches = 350
lr0 = (0.016 / 256) * batchsize
epoch_per_eval = 5
epoch_per_ckpt = 40
ema_alpha = 0.9999


def set_model():
    model = EfficientNet(model_type=model_type, n_classes=1000)
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, )):
            nn.init.kaiming_normal_(
                mod.weight,
                a=0,
                mode='fan_out',
                nonlinearity='leaky_relu')
            if not mod.bias is None: nn.init.constant_(mod.bias, 0)
        elif isinstance(mod, (nn.Linear, )):
            nn.init.xavier_normal_(mod.weight)
            if not mod.bias is None: nn.init.constant_(mod.bias, 0)
    model.cuda()
    ## TODO: official use label smooth
    criteria = nn.CrossEntropyLoss()
    criteria.cuda()
    return model, criteria


def set_dataloader(cropsize):
    dstrain = ImageNet(root='./imagenet', mode='train', cropsize=cropsize)
    dsval = ImageNet(root='./imagenet', mode='val', cropsize=cropsize)
    sampler_train = torch.utils.data.distributed.DistributedSampler(dstrain)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dsval)
    dltrain = torch.utils.data.DataLoader(
        dstrain,
        sampler=sampler_train,
        batch_size=bs_per_gpu,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    dlval = torch.utils.data.DataLoader(
        dsval,
        shuffle=False,
        sampler=sampler_val,
        batch_size=bs_per_gpu,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    return dltrain, dlval


def set_optimizer(model):
    bn_params, non_bn_params = [], []
    for name, param in model.named_parameters():
        if 'bn' in name:
            bn_params.append(param)
        else:
            non_bn_params.append(param)
    params_list = [
        {'params': non_bn_params},
        {'params': bn_params, 'weight_decay': 0},
    ]
    #  optimizer = torch.optim.SGD(
    #      params_list,
    #      lr=lr0,
    #      weight_decay=1e-5,
    #      momentum=0.9
    #  )
    optimizer = torch.optim.RMSprop(
        params_list,
        lr=lr0,
        alpha=0.9,
        eps=1e-3,
        weight_decay=1e-5,
        momentum=0.9
    )
    # TODO: chack implementation of this curve
    n_iters_per_epoch = n_train_imgs // batchsize
    lr_scheduler = WarmupExpLrScheduler(
        optimizer,
        power=0.97,
        step_interval=n_iters_per_epoch * 2.4,
        warmup_iter=n_iters_per_epoch * 5,
        warmup='exp',
        ## TODO: check warmup start ratio in the official
        warmup_ratio=1e-2
    )
    ema = EMA(model, ema_alpha)
    return optimizer, lr_scheduler, ema

def set_fp16(model, optimizer):
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    local_rank = dist.get_rank()
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank, ], output_device=local_rank)
    return model, optimizer


def set_meters():
    n_iters = n_epoches * n_train_imgs // batchsize
    time_meter = TimeMeter(n_iters)
    loss_meter = AvgMeter()
    return time_meter, loss_meter


def train(
        ep,
        model,
        criteria,
        dltrain,
        optimizer,
        lr_schdlr,
        ema,
        time_meter,
        loss_meter):
    model.train()
    dltrain.sampler.set_epoch(ep)
    for it, (ims, lbs) in enumerate(dltrain):
        ims = ims.cuda()
        lbs = lbs.cuda()

        logits = model(ims)
        loss = criteria(logits, lbs)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        lr_schdlr.step()
        #  ema.update_params()

        time_meter.update()
        loss_meter.update(loss.item())

        if (it+1) % 100 == 0:
            t_intv, eta = time_meter.get()
            loss_avg, _ = loss_meter.get()
            lr = ['{:.4f}'.format(el) for el in lr_schdlr.get_lr()]
            msg = 'epoch: {}, iter: {}, loss: {:.4f}, lr: {}, time: {:.2f}, eta: {}'.format(
                ep+1, it+1, loss_avg, lr, t_intv, eta
            )
            if dist.get_rank() == 0: print(msg)
    #  ema.update_buffer()


def evaluate(ema, dlval):
    org_states = ema.model.state_dict()
    ema.model.load_state_dict(ema.state)
    acc1, acc5 = eval_model(ema.model, dlval)
    ema.model.load_state_dict(org_states)
    return acc1, acc5


def eval_model(model, dlval):
    matches = []
    model.eval()
    with torch.no_grad():
        for _, (ims, lbs) in enumerate(dlval):
            ims = ims.cuda()
            lbs = lbs.cuda()

            logits = model(ims)
            scores = torch.softmax(logits, dim=1)
            rank_preds = torch.argsort(-scores, dim=1)
            match = (rank_preds == lbs.unsqueeze(1))
            matches.append(match)
    matches = torch.cat(matches, dim=0).float()
    n_correct_rank1 = matches[:, :1].float().sum().cuda()
    n_correct_rank5 = matches[:, :1].float().sum().cuda()
    n_samples = torch.tensor(matches.size(0)).float().cuda()
    dist.all_reduce(n_correct_rank1, dist.ReduceOp.SUM)
    dist.all_reduce(n_correct_rank5, dist.ReduceOp.SUM)
    dist.all_reduce(n_samples, dist.ReduceOp.SUM)
    acc_rank1 = n_correct_rank1 / n_samples
    acc_rank5 = n_correct_rank5 / n_samples

    return acc_rank1.item(), acc_rank5.item()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank',
                       dest='local_rank',
                       type=int,
                       default=-1,)
    parse.add_argument('--ckpt',
                       dest='ckpt',
                       type=str,
                       default=None,)
    return parse.parse_args()


def init_dist(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )


def main():
    args = parse_args()
    init_dist(args)

    model, criteria = set_model()

    optimizer, lr_schdlr, ema = set_optimizer(model)

    model, optimizer = set_fp16(model, optimizer)

    dltrain, dlval = set_dataloader(cropsize=cropsize)

    time_meter, loss_meter = set_meters()

    ep_start = 0
    if args.ckpt is not None:
        state = load_ckpt(args.ckpt, model, optimizer, lr_schdlr, ema)
        model, optimizer, lr_schdlr, ema, ep_start = state

    for ep in range(ep_start, n_epoches):
        train(
            ep,
            model,
            criteria,
            dltrain,
            optimizer,
            lr_schdlr,
            ema,
            time_meter,
            loss_meter)
        if (ep+1) % epoch_per_eval == 0:
            ## TODO: ema evaluation, 官方连bn的buffer参数都给ema了
            if dist.get_rank() == 0: print('eval model ...')
            #  acc1, acc5 = evaluate(ema, dlval)
            acc1, acc5 = eval_model(model, dlval)
            if dist.get_rank() == 0:
                print('epoch: {}, acc1: {:.4f}, acc5: {:.4f}'.format(ep+1, acc1, acc5))
        if (ep+1) % epoch_per_ckpt == 0 and dist.get_rank() == 0:
            ckpt_pth = './output/checkpoint_{}.pth'.format(ep)
            save_ckpt(ckpt_pth, ep, model, lr_schdlr, optimizer, ema)

    if dist.get_rank() == 0:
        ema.save_model('./output/final_model.pth')


if __name__ == '__main__':
    main()
