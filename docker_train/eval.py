import os.path as osp
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

from cbl_models import build_model
from config import set_cfg_from_file
from data import get_dataset

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader


#  from config.effnetb0 import *
#  from config.effnetb2 import *
#  from config.effnetb0_lite import *
#  from config.effnetb2_lite import *
#  from config.resnet101 import *
#  from config.frelu_resnet101 import *
#  from config.ibn_a_resnet50 import *
#  from config.ibn_b_resnet101_blur import *
#  from config.repvgg_a1 import *
#  from config.repvgg_a2 import *




def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank',
                       dest='local_rank',
                       type=int,
                       default=-1,)
    parse.add_argument('--config', dest='config', type=str, default='resnet50.py',)
    parse.add_argument('--ckpt', dest='ckpt_path', type=str, default='./res/model_final_naive.pth',)
    return parse.parse_args()




def evaluate(cfg, model, dl_eval):
    metric_dict = eval_model(model, dl_eval, cfg.metric)
    torch.cuda.empty_cache()
    return metric_dict


@torch.no_grad()
def eval_model(model, dl_eval, metric='acc'):
    model.eval()
    all_scores, all_gts = [], []
    for idx, (im, lb) in enumerate(dl_eval):
        im = im.cuda()
        lb = lb.cuda()
        #  lb = torch.zeros_like(lb)
        logits = model(im)
        if logits.size(1) == 1:
            scores = logits.sigmoid() # score: (bs, 1), lb: (bs, 1)
        else:
            scores = torch.softmax(logits, dim=1)

        all_scores.append(scores)
        all_gts.append(lb)
    all_scores = torch.cat(all_scores, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    metric_dict = {}
    if metric == 'acc':
        acc1, acc5 = compute_acc(all_scores, all_gts)
        metric_dict['acc1'] = acc1
        metric_dict['acc5'] = acc5
    elif metric == 'roc_auc':
        roc_auc = compute_roc_auc_score(all_scores, all_gts)
        metric_dict['roc_auc'] = roc_auc
    else:
        raise NotImplementedError
    return metric_dict


@torch.no_grad()
def compute_acc(all_scores, all_gts):
    if all_scores.size(1) == 1: # binary
        all_preds = all_scores.round().long()
        n_correct = (all_gts == all_preds).sum()
        n_samples = torch.tensor(all_gts.size(0)).cuda()
        if dist.is_initialized():
            dist.all_reduce(n_correct, dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, dist.ReduceOp.SUM)
        acc1 = n_correct / n_samples
        acc5 = 1.
    else: # multi-class
        all_preds = torch.argsort(-all_scores, dim=1)
        match = (all_preds == all_gts.unsqueeze(1)).float()
        n_correct_1 = match[:, :1].sum()
        n_correct_5 = match[:, :5].sum()
        n_samples = torch.tensor(match.size(0)).cuda()
        if dist.is_initialized():
            dist.all_reduce(n_correct_1, dist.ReduceOp.SUM)
            dist.all_reduce(n_correct_5, dist.ReduceOp.SUM)
            dist.all_reduce(n_samples, dist.ReduceOp.SUM)
        acc1 = (n_correct_1 / n_samples).item()
        acc5 = (n_correct_5 / n_samples).item()
    return acc1, acc5


@torch.no_grad()
def compute_roc_auc_score(all_scores, all_gts):
    if not all_scores.size(1) == 1: raise NotImplementedError

    all_scores = all_scores.contiguous()
    all_gt_gather = [torch.ones_like(all_gts)
            for _ in range(dist.get_world_size())]
    all_scores_gather = [torch.ones_like(all_scores)
            for _ in range(dist.get_world_size())]
    dist.all_gather(all_gt_gather, all_gts, async_op=False)
    dist.all_gather(all_scores_gather, all_scores, async_op=False)
    all_gt_gather = torch.cat(all_gt_gather, dim=0).squeeze()
    all_scores_gather = torch.cat(all_scores_gather, dim=0).squeeze()
    all_gt_gather = all_gt_gather.detach().cpu().numpy()
    all_scores_gather = all_scores_gather.detach().cpu().numpy()
    roc_auc = roc_auc_score(y_true=all_gt_gather, y_score=all_scores_gather)
    return roc_auc


def main():
    args = parse_args()
    cfg = set_cfg_from_file(args.config)

    init_dist(args)

    model = build_model(cfg.model_args)
    sd = torch.load(args.ckpt_path, map_location='cpu')
    #  new_sd = {}
    #  for k, v in sd.items():
    #      k = k.replace('module.', '')
    #      new_sd[k] = v
    #  model.load_state_dict(new_sd)
    #  model.load_state_dict(torch.load('./res/model_final.pth', map_location='cpu'))
    #  model.load_state_dict(sd, strict=True)
    model.load_states(sd)
    model.eval()
    if hasattr(model, 'fuse_block'):
        model.fuse_block()
    if hasattr(model, 'fuse_conv_bn'):
        model.fuse_conv_bn()
    model.cuda()
    #  if dist.get_rank() == 0:
    #      print(model)

    ds = get_dataset(cfg.dataset_args, mode='val')
    if dist.is_initialized():
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=False)
        batch_sampler_val = torch.utils.data.sampler.BatchSampler(
            sampler_val, cfg.batchsize, drop_last=False
        )
        dl = DataLoader(
            ds, batch_sampler=batch_sampler_val,
            num_workers=8, pin_memory=True
        )
    else:
        dl = DataLoader(
            ds, batch_size=256, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=False
        )

    metric_dict = evaluate(cfg, model, dl)

    if not (dist.is_initialized() and dist.get_rank() != 0):
        msg = f'eval result: {metric_dict}'
        print(msg)


def init_dist(args):
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )

if __name__ == "__main__":
    main()
