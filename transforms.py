
import random
import cv2
import numpy as np
import torch


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        return cv2.resize(im, self.size)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im):
        return im[:, ::-1, :]


class RandomCrop(object):
    def __init__(self, cropsize):
        self.cropsize = cropsize

    def __call__(self, im):
        H, W, _ = im.shape
        cH, cW = self.cropsize
        assert H > self.cropsize[0] and W > self.cropsize[1]
        mH, mW = H - cH, W - cW
        h, w = random.randint(0, mH), random.randint(0, mW)
        im = im[h:h+cH, w:w+cW, :]
        return im


class Pad(object):
    def __init__(self, padding, fill=0):
        self.padding = padding
        self.fill = fill

    def __call__(self, im):
        top = bottom = self.padding[0]
        left = right = self.padding[1]
        return cv2.copyMakeBorder(
            im, top=top, bottom=bottom, left=left, right=right,
            borderType=cv2.BORDER_CONSTANT, value=self.fill
        )


class ChannelDrop(object):
    def __init__(self, n_max_chan_drop=2):
        self.n_max_chan_drop = n_max_chan_drop

    def __call__(self, im):
        n_drops = random.randint(0, self.n_max_chan_drop)
        if n_drops == 0:
            return im
        drop_chan = random.sample([0, 1, 2], n_drops)
        im = np.array(im)
        im[:, :, drop_chan] = 0
        return im


class ColorJitter(object):
    '''
    There is no opencv way to do it, so let convert it to PIL and use PIL to do it
    '''
    def __init__(self):
        pass

    def __call__(self, im):
        pass


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, im):
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        if isinstance(im, torch.ByteTensor) or im.dtype == torch.uint8:
            im = im.float().div(255)
        return im


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, im):
        _ = [ten.sub_(m).div_(s) for ten, m, s in zip(im, self.mean, self.std)]
        return im


class Compose(object):
    def __init__(self, trans):
        self.transforms = trans

    def __call__(self, im):
        for trans in self.transforms:
            im = trans(im)
        return im


if __name__ == '__main__':
    im_org = cv2.imread('pic.jpg')
    cv2.imshow('org', im_org)
    cv2.waitKey(0)

    #  im = Resize((32, 32))(im_org)
    #  cv2.imshow('resize', im)
    #  cv2.waitKey(0)

    #  im = RandomHorizontalFlip(p=0.5)(im_org)
    #  cv2.imshow('h flip', im)
    #  cv2.waitKey(0)
    #
    #  im = RandomCrop(cropsize=(45, 45))(im_org)
    #  cv2.imshow('random crop', im)
    #  cv2.waitKey(0)

    #  im = ChannelDrop()(im_org)
    #  cv2.imshow('color drop', im)
    #  cv2.waitKey(0)

    im = Pad((10, 10), fill=180)(im_org)
    cv2.imshow('pad', im)
    cv2.waitKey(0)
