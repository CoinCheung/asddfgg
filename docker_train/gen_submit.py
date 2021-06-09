
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
        './config/seti/resnet50.py',
        ]
ckpt_paths = [
        './res/model_final_ema.pth',
        ]
batchsize = 32



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
            A.Resize(p=1., height=320, width=320),
            ToTensorV2(),
        ])

    def __getitem__(self, ind):
        fid = self.fids[ind]
        pth = osp.join(test_data_root, fid_pths[fid])
        im = np.load(pth)[[0, 2, 4]] # (3, 273, 256)
        im = np.vstack(im) # (819, 256)
        im = im.T.astype('f')[..., np.newaxis] # (256, 819, 1)
        im = self.trans(image=im)['image']
        return im, fid

    def __len__(self,):
        return len(fid_pths)


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
            logits = model(imgs)
            if logits.size(1) == 1:
                scores += logits.sigmoid().squeeze()
            else:
                scores += logits.softmax(dim=1)[:, 1]
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
