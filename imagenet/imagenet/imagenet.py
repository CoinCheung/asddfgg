import os
import os.path as osp
import json

from torch.utils.data import Dataset, DataLoader

import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
#  import transforms as T
from autoaugment import ImageNetPolicy


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
        self.trans_train = T.Compose([
            T.RandomResizedCrop(cropsize),
            T.RandomHorizontalFlip(),
            ImageNetPolicy(),
        ])
        self.trans_val = T.Compose([
            T.Resize((cropsize, cropsize)),
        ])
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


    def __getitem__(self, idx):
        impth, label = self.samples[idx]
        im = Image.open(impth).convert('RGB')
        if self.mode == 'train':
            im = self.trans_train(im)
        else:
            im = self.trans_val(im)
        im = self.to_tensor(im)
        return im, label

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    # 10
    ds = ImageNet(root='./data', mode='train')
    im, lb = ds[40948]
    print(im.size())
    print(lb)
    #  dltrain = torch.utils.data.DataLoader(
    #      ds,
    #      shuffle=True,
    #      batch_size=256,
    #      num_workers=4,
    #      pin_memory=True,
    #  )
    #  for ims, lbs in dltrain:
    #      print(ims.size())
    #      print(lbs.size())
