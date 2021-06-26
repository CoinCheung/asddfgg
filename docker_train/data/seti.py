import os
import os.path as osp
import json
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import transforms as T

from rand_augment_cv2 import RandomAugment


class SETI(Dataset):

    def __init__(self, root='./', mode='train', cropsize=224, binary=True):
        super(SETI, self).__init__()
        self.samples = []
        self.mode = mode
        self.cropsize = cropsize
        self.binary = binary
        if mode == 'train':
            txtpth = osp.join(root, 'train.txt')
        elif mode == 'val':
            txtpth = osp.join(root, 'val.txt')
        with open(txtpth, 'r') as fr:
            lines = fr.read().splitlines()
        for line in lines:
            pth, lb = line.split(',')
            pth, lb = osp.join(root, pth), int(lb)
            self.samples.append((pth, lb))

        self.trans_train = A.Compose([
            A.Resize(p=1., height=cropsize, width=cropsize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5, shift_limit_x=(-0.2, 0.2),
                shift_limit_y=(-0.2, 0.2), scale_limit=(-0.20, 0.20),
                rotate_limit=(-20, 20), interpolation=1, border_mode=0,
                value=0, mask_value=0),
            A.RandomResizedCrop(p=1.0, width=cropsize,
                height=cropsize, scale=(0.9, 1.0)),
            A.Normalize(mean=(0.4),std=(0.2),max_pixel_value=180.,p=1.),
            ToTensorV2(),
            #  RandomAugment(2, 9),
            #  T.PCANoise(0.1),
        ])
        self.trans_val = A.Compose([
            A.Resize(p=1., height=cropsize, width=cropsize),
            A.Normalize(mean=(0.4),std=(0.2),max_pixel_value=180.,p=1.),
            ToTensorV2(),
        ])

    #  def readimg(self, impth):
    #      im = np.load(impth)[[0, 2, 4]] # (3, 273, 256)
    #      im = np.vstack(im) # (819, 256)
    #      im = im.T.astype('f')[..., np.newaxis] # (256, 819, 1)
    #      return im

    def readimg(self, impth):
        im_on = np.load(impth)[[0, 2, 4]] # (3, 273, 256)
        im_off = np.load(impth)[[1, 3, 5]] # (3, 273, 256)
        im_on = np.vstack(im_on).T[..., np.newaxis] # (256, 819, 1)
        im_off = np.vstack(im_off).T[..., np.newaxis] # (256, 819, 1)
        im = np.concatenate([im_on, im_off], axis=2)
        im = im.astype('f') # (256, 819, 2)
        return im

    def __getitem__(self, idx):
        impth, label = self.samples[idx]
        im = self.readimg(impth)
        if self.mode == 'train':
            im = self.trans_train(image=im)
        else:
            im = self.trans_val(image=im)
        im = im['image']
        if self.binary: label = torch.tensor([label]).float()
        return im, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    pass
