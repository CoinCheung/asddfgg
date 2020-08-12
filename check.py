import os
import os.path as osp
import json

#  root = './imagenet/data'
#
#  train_samples = []
#  with open('./imagenet/utils/imagenet_class_index.json', 'r') as fr:
#      jobj = json.load(fr)
#  cls_to_id = {v[0]: int(k) for k, v in jobj.items()}
#  trainpth = osp.join(root, 'train')
#  folders = os.listdir(trainpth)
#  for fd in folders:
#      fdpth = osp.join(trainpth, fd)
#      if osp.isfile(fdpth): continue
#      imgs = [osp.join(fdpth, el) for el in os.listdir(fdpth)]
#      label = cls_to_id[fd]
#      train_samples += [(el, label) for el in imgs]
#
#
#
#  val_samples = []
#  with open('./imagenet/utils/ILSVRC2012_validation_ground_truth.txt', 'r') as fr:
#      labels = [int(el) for el in fr.read().splitlines()]
#  valpth = osp.join(root, 'val')
#  imgs = os.listdir(valpth)
#  idx_imgs = {
#      int(el.split('.')[0].split('_')[-1])-1: osp.join(valpth, el)
#      for el in imgs
#  }
#  val_samples = [
#      (impth, labels[idx]) for idx, impth in idx_imgs.items()
#  ]
#
#
#  val = val_samples[0]
#  val_pth, val_lb = val
#
#  for sp in train_samples:
#      train = sp
#      train_pth, train_lb = sp
#      if train_lb == val_lb:
#          print(train_lb)
#          print(train_pth)
#          print(val_pth)
#          break
#

import torch

a = torch.randn(4, 3, 10, 10)
x1 = torch.randint(0, 5, (4,))
y1 = torch.randint(0, 5, (4,))
x2 = torch.randint(6, 10, (4,))
y2 = torch.randint(6, 10, (4,))

idx = torch.randint(0, 2, (2,))
print(a[torch.arange(2), :, idx, idx].size())

#  print(x1)
#  print(x2)
#  print(a[x1])
#  print(a[:])
#  b = a[torch.arange(4), :, x1:x2, y1:y2]
#  print(b)
