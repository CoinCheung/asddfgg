#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d
#  from .conv_ops import Conv2dWS as Conv2d

from .resnet_base import ResNetBackboneBase, ResNetBase, ResNetDenseCLBase



## resnet-v1
class ResNetBackbone(ResNetBackboneBase):

    def __init__(self, in_chan=3, n_layers=50, stride=32, use_blur_pool=False):
        super(ResNetBackbone, self).__init__(
                in_chan=in_chan, n_layers=n_layers, stride=stride,
                use_blur_pool=use_blur_pool
                )



class ResNet(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000, use_blur_pool=False):
        super(ResNet, self).__init__()
        self.backbone = ResNetBackbone(n_layers=n_layers, use_blur_pool=use_blur_pool)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


## se-resnet-v1
class SEResNetBackbone(ResNetBackboneBase):

    def __init__(self, n_layers=50, stride=32):
        super(SEResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride, use_se=True)


class SEResNet(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(SEResNet, self).__init__()
        self.backbone = SEResNetBackbone(n_layers, stride)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


## askc-resnet-v1
class ASKCResNetBackbone(ResNetBackboneBase):

    def __init__(self, n_layers=50, stride=32):
        super(ASKCResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride, use_askc=True)


class ASKCResNet(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(ASKCResNet, self).__init__()
        self.backbone = ASKCResNetBackbone(n_layers, stride)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


## weight-align-resnet-v1
class WAResNetBackbone(ResNetBackboneBase):

    def __init__(self, n_layers=50, stride=32):
        super(WAResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride, use_askc=False, conv_type='wa')


class WAResNet(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(WAResNet, self).__init__()
        self.backbone = WAResNetBackbone(n_layers, stride)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


## frelu-resnet-v1
class FReLUResNetBackbone(ResNetBackboneBase):

    def __init__(self, n_layers=50, stride=32):
        super(FReLUResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride, use_askc=False, conv_type='nn',
                act_type='frelu')


class FReLUResNet(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(FReLUResNet, self).__init__()
        self.backbone = FReLUResNetBackbone(n_layers, stride)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


## dy_conv-resnet-v1
class DYConvResNetBackbone(ResNetBackboneBase):

    def __init__(self, n_layers=50, stride=32):
        super(DYConvResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride, use_askc=False, conv_type='dy',
                act_type='relu')


class DYConvResNet(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(DYConvResNet, self).__init__()
        self.backbone = DYConvResNetBackbone(n_layers, stride)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


## ibn-a-resnet-v1
class IBNResNetBackboneA(ResNetBackboneBase):

    def __init__(self, in_chan=3, n_layers=50, stride=32, use_blur_pool=False):
        super(IBNResNetBackboneA, self).__init__(
                in_chan=in_chan, n_layers=n_layers, stride=stride, ibn='a', act_type='relu',
                use_blur_pool=use_blur_pool
                )


class IBNResNetA(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000, use_blur_pool=False):
        super(IBNResNetA, self).__init__()
        self.backbone = IBNResNetBackboneA(n_layers=n_layers, use_blur_pool=use_blur_pool)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


class IBNResNetDenseCLA(ResNetDenseCLBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(IBNResNetDenseCLA, self).__init__(dim=2048, n_classes=n_classes)
        self.backbone = IBNResNetBackboneA(n_layers, stride)


## ibn-b-resnet-v1
class IBNResNetBackboneB(ResNetBackboneBase):

    def __init__(self, in_chan=3, n_layers=50, stride=32, use_blur_pool=False):
        super(IBNResNetBackboneB, self).__init__(
                in_chan=in_chan, n_layers=n_layers, stride=stride, ibn='b', act_type='relu',
                use_blur_pool=use_blur_pool
                )


class IBNResNetB(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000, use_blur_pool=False):
        super(IBNResNetB, self).__init__()
        self.backbone = IBNResNetBackboneB(n_layers=n_layers, use_blur_pool=use_blur_pool)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


class IBNResNetDenseCLB(ResNetDenseCLBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(IBNResNetDenseCLB, self).__init__(dim=2048, n_classes=n_classes)
        self.backbone = IBNResNetBackboneB(n_layers, stride)





if __name__ == "__main__":
    #  layer1 = create_stage(64, 256, 3, 1, 1)
    #  layer2 = create_stage(256, 512, 4, 2, 1)
    #  layer3 = create_stage(512, 1024, 6, 1, 2)
    #  layer4 = create_stage(1024, 2048, 3, 1, 4)
    #  print(layer4)
    resnet = IBNResNetA()
    inten = torch.randn(1, 3, 224+33, 224)
    out = resnet(inten)
    print(out.size())
    import torchvision
    model = torchvision.models.resnet50(pretrained=False)
    #  for name, param in resnet.named_parameters():
    #      if 'bias' in name:
    #          print(name)


