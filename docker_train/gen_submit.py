
import os
import os.path as osp
import argparse

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from cbl_models import build_model
from config import set_cfg_from_file

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader


torch.set_grad_enabled(False)


test_data_root = 'datasets/seti'
test_ann_file = 'datasets/seti/test.txt'
cfg_files = [
        #  './config/seti/resnet50_adamw_warmup10.py',
        #  './config/seti/resnet50_adamw_warmup10.py',
        #  './config/seti/resnet50_adamw_warmup10.py',
        #  './config/seti/resnet50_adamw_warmup10.py',
        #  './config/seti/resnet50_adamw_warmup10.py',
        './config/seti/timm_r18d.py',
        './config/seti/timm_r18d.py',
        './config/seti/timm_r18d.py',
        './config/seti/timm_r18d.py',
        './config/seti/timm_r18d.py',
        ]
ckpt_paths = [
        #  './res/model_final_naive.pth',
        './res/model_final_naive_1.pth',
        './res/model_final_naive_2.pth',
        './res/model_final_naive_3.pth',
        './res/model_final_naive_4.pth',
        './res/model_final_naive_5.pth',
        ]
batchsize = 32
cropsize = 512
hflip = False
vflip = False
hvflip = False



def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank',
                       dest='local_rank',
                       type=int,
                       default=-1,)
    return parse.parse_args()



with open(test_ann_file, 'r') as fr:
    lines = fr.read().splitlines()

fid_pths = {}
for pth in lines:
    fid = osp.splitext(osp.basename(pth))[0]
    fid_pths[fid] = pth


def define_models():
    models, cfgs = [], []
    for cfg_file, ckpt_pth in zip(cfg_files, ckpt_paths):
        cfg = set_cfg_from_file(cfg_file)
        model = build_model(cfg.model_args)
        state = torch.load(ckpt_pth, map_location='cpu')
        model.load_states(state)
        model.cuda().eval()
        models.append(model)
        cfgs.append(cfg)
    return models, cfgs


class InferDataset(Dataset):

    def __init__(self):
        super(InferDataset, self).__init__()
        self.fids = list(fid_pths.keys())
        self.trans = A.Compose([
            A.Resize(p=1., height=cropsize, width=cropsize),
            A.Normalize(mean=(0.4),std=(0.2),max_pixel_value=180.,p=1.),
            ToTensorV2(),
        ])

    def readimg(self, impth):
        im_on = np.load(impth)[[0, 2, 4]] # (3, 273, 256)
        im_off = np.load(impth)[[1, 3, 5]] # (3, 273, 256)
        im_on = np.vstack(im_on).T[..., np.newaxis] # (256, 819, 1)
        im_off = np.vstack(im_off).T[..., np.newaxis] # (256, 819, 1)
        im = np.concatenate([im_on, im_off], axis=2)
        im = im.astype('f') # (256, 819, 2)
        return im

    #  def readimg(self, impth):
    #      im = np.load(impth)[[0, 2, 4]] # (3, 273, 256)
    #      im = np.vstack(im) # (819, 256)
    #      im = im.T.astype('f')[..., np.newaxis] # (256, 819, 1)
    #      return im

    def __getitem__(self, ind):
        fid = self.fids[ind]
        pth = osp.join(test_data_root, fid_pths[fid])
        im = self.readimg(pth)
        im = self.trans(image=im)['image']
        return im, fid

    def __len__(self,):
        return len(fid_pths)


def infer_with_model(model, imgs):
    bs = imgs.size(0)
    scores = torch.zeros(bs).cuda()
    logits = model(imgs)
    if logits.size(1) == 1:
        scores += logits.sigmoid().squeeze()
    else:
        scores += logits.softmax(dim=1)[:, 1]
    return scores


def run_submit():
    # models
    models, cfgs = define_models()

    # dataloader
    ds = InferDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=False)
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batchsize, drop_last=False)
    dl = DataLoader(
        #  ds, batch_size=batchsize, shuffle=False, drop_last=False,
        ds, batch_sampler=batch_sampler,
        num_workers=4, pin_memory=True
    )
    local_rank = dist.get_rank()
    dl = tqdm(dl) if local_rank == 0 else dl

    # infer
    lines = []
    for imgs, fids in dl:
        imgs = imgs.cuda()
        bs = imgs.size(0)
        scores = torch.zeros(bs).cuda()
        n_adds = 0
        for model in models:
            #  logits = model(imgs)
            #  if logits.size(1) == 1:
            #      scores += logits.sigmoid().squeeze()
            #  else:
            #      scores += logits.softmax(dim=1)[:, 1]
            scores += infer_with_model(model, imgs)
            n_adds += 1
            if hflip:
                scores += infer_with_model(model, imgs.flip(dims=(3,)))
                n_adds += 1
            if vflip:
                scores += infer_with_model(model, imgs.flip(dims=(2,)))
                n_adds += 1
            if hvflip:
                scores += infer_with_model(model, imgs.flip(dims=(2, 3)))
                n_adds += 1
        scores /= n_adds
        scores = scores.tolist()
        for fid, score in zip(fids, scores):
            lines.append(f'{fid},{score}')

    with open(f'./res/submit_worker_{local_rank}.csv', 'w') as fw:
        fw.write('\n'.join(lines))

    with open(f'./res/worker_{local_rank}_done', 'w') as fw:
        fw.write('')


    if local_rank == 0:
        ## synchronize
        while True:
            n_done = 0
            for rk in range(torch.cuda.device_count()):
                if osp.exists(f'./res/worker_{rk}_done'):
                    n_done += 1
            if n_done == torch.cuda.device_count(): break

        ## merge results
        lines = []
        for rk in range(torch.cuda.device_count()):
            with open(f'./res/submit_worker_{rk}.csv', 'r') as fr:
                lines += fr.read().splitlines()
        res_dict = {}
        for line in lines:
            fid, score = line.split(',')
            res_dict[fid] = score
        lines = ['id,target',] + [f'{k},{v}' for k,v in res_dict.items()]
        with open(f'./res/submit.csv', 'w') as fw:
            fw.write('\n'.join(lines))


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
    run_submit()
