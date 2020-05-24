import os.path as osp
import argparse
import numpy as np

from resnet import ResNet50
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from imagenet.imagenet_cv2 import ImageNet



def evaluate(model, dl_eval):
    acc_1, acc_5 = eval_model(model, dl_eval)
    torch.cuda.empty_cache()
    return acc_1, acc_5

#  def evaluate(ema, dl_eval):
#      ema.apply_shadow()
#      acc_1_ema, acc_5_ema = eval_model(ema.model, dl_eval)
#      ema.restore()
#      acc_1, acc_5 = eval_model(ema.model, dl_eval)
#      torch.cuda.empty_cache()
#      return acc_1, acc_5, acc_1_ema, acc_5_ema


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
    model = ResNet50()
    sd = torch.load('./res/model_final.pth', map_location='cpu')
    new_sd = {}
    for k, v in sd.items():
        k = k.replace('module.', '')
        new_sd[k] = v
    model.load_state_dict(new_sd)
    #  model.load_state_dict(torch.load('./res/model_final.pth', map_location='cpu'))
    model.cuda()

    batchsize = 256
    ds = ImageNet('./imagenet', 'val')
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

    #  import torchvision.datasets as datasets
    #  import torchvision.transforms as transforms
    #  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                   std=[0.229, 0.224, 0.225])
    #  valdir = osp.join('/data1/zzy/datasets/imagenet', 'val_cls')
    #  dl = torch.utils.data.DataLoader(
    #      datasets.ImageFolder(valdir, transforms.Compose([
    #          transforms.Resize(256),
    #          transforms.CenterCrop(224),
    #          transforms.ToTensor(),
    #          normalize,
    #      ])),
    #      batch_size=256, shuffle=False,
    #      num_workers=8, pin_memory=True)
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
