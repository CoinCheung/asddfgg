import random
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T


class ChannelDrop(object):
    def __init__(self):
        pass

    def __call__(self, im):
        n_drops = random.randint(0, 2)
        if n_drops == 0:
            return im
        drop_chan = random.sample([0, 1, 2], n_drops)
        im = np.array(im)
        im[:, :, drop_chan] = 0
        im = Image.fromarray(im)
        return im


def get_train_loader(batch_size, num_workers, dataset='cifar10', pin_memory=True):
    assert dataset in ('cifar10', 'cifar100'), 'unrecognised dataset'
    trans = T.Compose([
        #  ChannelDrop(),
        T.Resize((36, 36)),
        T.RandomCrop((32, 32)),
        T.RandomHorizontalFlip(0.5),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10(
            root='./dataset/cifar-10-batches-py',
            train=True,
            transform=trans,
            download=True,
        )
    else:
        ds = torchvision.datasets.CIFAR100(
            root='./dataset/cifar-100-batches-py',
            train=True,
            transform=trans,
            download=True,
        )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


def get_val_loader(batch_size, num_workers, dataset='cifar10', pin_memory=True):
    assert dataset in ('cifar10', 'cifar100'), 'unrecognised dataset'
    trans = T.Compose([
        T.Resize((32, 32)),
        T.RandomCrop((32, 32)),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    if dataset == 'cifar10':
        ds = torchvision.datasets.CIFAR10(
            root='./dataset/cifar-10-batches-py',
            train=False,
            transform=trans,
            download=False,
        )
    else:
        ds = torchvision.datasets.CIFAR100(
            root='./dataset/cifar-100-batches-py',
            train=False,
            transform=trans,
            download=False,
        )
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return dl


if __name__ == "__main__":
    # 10
    dltrain = get_train_loader(16, 4, 'cifar10', False)
    for im, lb in dltrain:
        print(im.size())
        print(lb.size())
        break

    dlval = get_val_loader(16, 4, 'cifar10', False)
    for im, lb in dlval:
        print(im.size())
        print(lb.size())
        break
    print(len(dltrain))
    print(len(dlval))

    # 100
    dltrain = get_train_loader(16, 4, 'cifar100', False)
    for im, lb in dltrain:
        print(im.size())
        print(lb.size())
        break

    dlval = get_val_loader(16, 4, 'cifar100', False)
    for im, lb in dlval:
        print(im.size())
        print(lb.size())
        break
    print(len(dltrain))
    print(len(dlval))
