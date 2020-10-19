import os
import os.path as osp
import json
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import transforms as T

from rand_augment_cv2 import RandomAugment


class ImageNet(Dataset):

    def __init__(self, root='./', mode='train', cropsize=224):
        super(ImageNet, self).__init__()
        self.samples = []
        self.mode = mode
        self.cropsize = cropsize
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
        img_mean, img_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.trans_train = T.Compose([
            T.RandomResizedCrop(cropsize),
            T.RandomHorizontalFlip(),
            RandomAugment(2, 9),
            T.ToTensor(),
            T.PCANoise(0.1),
            T.Normalize(img_mean, img_std)
            #  T.ColorJitter(0.4, 0.4, 0.4),
        ])
        short_size = int(cropsize * 256 / 224)
        self.trans_val = T.Compose([
            T.ResizeCenterCrop(crop_size=cropsize, short_size=short_size),
            T.ToTensor(),
            T.Normalize(img_mean, img_std)
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
        else:
            im = self.trans_val(im)
        return im, label

    #  def __getitem__(self, idx):
    #      import cdataloader
    #      impth, label = self.samples[idx]
    #      cropsize = [self.cropsize, self.cropsize]
    #      pca_noies = 0.1
    #      is_train = True if self.mode == 'train' else False
    #      use_ra = False
    #      im = cdataloader.get_img_by_path(impth, cropsize, pca_noies, is_train, use_ra)
    #      return im, label

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
