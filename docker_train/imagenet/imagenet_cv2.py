import os
import os.path as osp
import json
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import transforms as T

from autoaugment import ImageNetPolicy
from rand_augment_cv2 import RandomAugment
from random_erasing import RandomErasing



class ImageNet(Dataset):

    def __init__(self, root='./', mode='train', cropsize=224):
        super(ImageNet, self).__init__()
        self.samples = []
        self.mode = mode
        if mode == 'train':
            txtpth = osp.join(root, 'train.txt')
            img_root_pth = osp.join(root, 'data', 'train')
            with open(txtpth, 'r') as fr:
                lines = fr.read().splitlines()
            for line in lines:
                pth, lb = line.split(' ')
                pth, lb = osp.join(img_root_pth, pth), int(lb)
                self.samples.append((pth, lb))
        elif mode == 'val':
            txtpth = osp.join(root, 'val.txt')
            img_root_pth = osp.join(root, 'data', 'val')
            with open(txtpth, 'r') as fr:
                lines = fr.read().splitlines()
            for line in lines:
                pth, lb = line.split(' ')
                pth, lb = osp.join(img_root_pth, pth), int(lb)
                self.samples.append((pth, lb))
        randaug_mean = tuple([min(255, int(x * 255)) for x in (0.485, 0.456, 0.406)])
        self.trans_train = T.Compose([
            T.RandomResizedCrop(cropsize),
            T.RandomHorizontalFlip(),
            RandomAugment(2, 9),
            #  rand_augment_transform('rand-m9-mstd0.5', {'translate_const': 100, 'img_mean': randaug_mean,}),
            T.ColorJitter(0.4, 0.4, 0.4),
        ])
        self.random_erasing = RandomErasing(0.2, mode='pixel', max_count=1)
        self.trans_val = T.Compose([
            T.ResizeCenterCrop((cropsize, cropsize)),
        ])
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def readimg(self, impth):
        im = cv2.imread(impth, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def __getitem__(self, idx):
        impth, label = self.samples[idx]
        im = self.readimg(impth)
        if self.mode == 'train':
            im = self.trans_train(im)
            im = self.to_tensor(im)
            im = self.random_erasing(im)
        else:
            im = self.trans_val(im)
            im = self.to_tensor(im)
        return im, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # 10
    ds = ImageNet(root='./imagenet', mode='train')
    #  im, lb = ds[40948]
    #  print(im.size())
    #  print(lb)
    dltrain = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=256,
        num_workers=4,
        pin_memory=True,
    )
    for ims, lbs in dltrain:
        print(ims.size())
        print(lbs.size())
