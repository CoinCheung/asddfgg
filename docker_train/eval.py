import os.path as osp
import argparse
import numpy as np

from cbl_models import build_model
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from imagenet.imagenet_cv2 import ImageNet

#  from config.effnetb0 import *
#  from config.effnetb2 import *
#  from config.effnetb0_lite import *
#  from config.effnetb2_lite import *
#  from config.resnet101 import *
#  from config.frelu_resnet101 import *
#  from config.ibn_a_resnet50 import *
#  from config.ibn_b_resnet101_blur import *
from config.repvgg_a2 import *


ckpt_path = './res/model_final_naive.pth'

def evaluate(model, dl_eval):
    acc_1, acc_5 = eval_model(model, dl_eval)
    torch.cuda.empty_cache()
    return acc_1, acc_5


@torch.no_grad()
def eval_model(model, dl_eval):
    model.eval()
    all_scores, all_gts = [], []
    for idx, (im, lb) in enumerate(dl_eval):
        im = im.cuda()
        lb = lb.cuda()
        logits = model(im)
        scores = torch.softmax(logits, dim=1)

        all_scores.append(scores)
        all_gts.append(lb)
    all_scores = torch.cat(all_scores, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
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
    torch.cuda.empty_cache()
    return acc1, acc5


def main():
    #  model = ResNet50()
    #  model = EfficientNet(model_type, n_classes)
    model = build_model(model_args)
    sd = torch.load(ckpt_path, map_location='cpu')
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

    batchsize = 256
    ds = ImageNet(datapth, mode='val', cropsize=cropsize)
    if dist.is_initialized():
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            ds, shuffle=False)
        batch_sampler_val = torch.utils.data.sampler.BatchSampler(
            sampler_val, batchsize, drop_last=False
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

    acc1, acc5 = evaluate(model, dl)

    if not (dist.is_initialized() and dist.get_rank() != 0):
        print('acc1: {}, acc5: {}'.format(acc1, acc5))


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

if __name__ == "__main__":
    args = parse_args()
    init_dist(args)
    main()
