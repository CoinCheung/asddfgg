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

    def __init__(self, root='./data', mode='train', cropsize=224):
        super(ImageNet, self).__init__()
        self.samples = []
        self.mode = mode
        if mode == 'train':
            with open('./utils/imagenet_class_index.json', 'r') as fr:
                jobj = json.load(fr)
            cls_to_id = {v[0]: int(k) for k, v in jobj.items()}
            trainpth = osp.join(root, 'train')
            folders = os.listdir(trainpth)
            for fd in folders:
                fdpth = osp.join(trainpth, fd)
                if osp.isfile(fdpth): continue
                imgs = [osp.join(fdpth, el) for el in os.listdir(fdpth)]
                label = cls_to_id[fd]
                self.samples += [(el, label) for el in imgs]
        elif mode == 'val':
            with open('./utils/ILSVRC2012_validation_ground_truth.txt', 'r') as fr:
                labels = [int(el) for el in fr.read().splitlines()]
            valpth = osp.join(root, 'val')
            imgs = os.listdir(valpth)
            idx_imgs = {
                int(el.split('.')[0].split('_')[-1])-1: osp.join(valpth, el)
                for el in imgs
            }
            self.samples = [
                (impth, labels[idx]) for idx, impth in idx_imgs.items()
            ]
        self.trans_train = T.Compose([
            T.RandomResizedCrop(cropsize),
            T.RandomHorizontalFlip(),
            ImageNetPolicy(),
        ])
        self.trans_val = T.Compose([
            T.Resize(cropsize),
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
    im, lb = ds[409548]
    print(im.size())
    print(lb)
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
