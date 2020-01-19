import os
import os.path as osp
import json

root = './imagenet/data'

train_samples = []
with open('./imagenet/utils/imagenet_class_index.json', 'r') as fr:
    jobj = json.load(fr)
cls_to_id = {v[0]: int(k) for k, v in jobj.items()}
trainpth = osp.join(root, 'train')
folders = os.listdir(trainpth)
for fd in folders:
    fdpth = osp.join(trainpth, fd)
    if osp.isfile(fdpth): continue
    imgs = [osp.join(fdpth, el) for el in os.listdir(fdpth)]
    label = cls_to_id[fd]
    train_samples += [(el, label) for el in imgs]



val_samples = []
with open('./imagenet/utils/ILSVRC2012_validation_ground_truth.txt', 'r') as fr:
    labels = [int(el) for el in fr.read().splitlines()]
valpth = osp.join(root, 'val')
imgs = os.listdir(valpth)
idx_imgs = {
    int(el.split('.')[0].split('_')[-1])-1: osp.join(valpth, el)
    for el in imgs
}
val_samples = [
    (impth, labels[idx]) for idx, impth in idx_imgs.items()
]


val = val_samples[0]
val_pth, val_lb = val

for sp in train_samples:
    train = sp
    train_pth, train_lb = sp
    if train_lb == val_lb:
        print(train_lb)
        print(train_pth)
        print(val_pth)
        break

