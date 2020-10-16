#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d
#  from .conv_ops import Conv2dWS as Conv2d

from .resnet_base import ResNetBackboneBase, ResNetBase
from .resnet_base import PAResNetBackBoneBase, PAResNetBase



## resnet-v1
class ResNetBackbone(ResNetBackboneBase):

    def __init__(self, n_layers=50, stride=32):
        super(ResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride)


class ResNet(ResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(ResNet, self).__init__()
        self.backbone = ResNetBackbone(n_layers, stride)
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


## resnet-v2
class PAResNetBackbone(PAResNetBackBoneBase):

    def __init__(self, n_layers=50, stride=32):
        super(PAResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride)


class PAResNet(PAResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(PAResNet, self).__init__()
        self.backbone = PAResNetBackbone(n_layers, stride)
        self.classifier = nn.Linear(2048, n_classes, bias=True)


## se-resnet-v2
class SE_PAResNetBackbone(PAResNetBackBoneBase):

    def __init__(self, n_layers=50, stride=32):
        super(SE_PAResNetBackbone, self).__init__(
                n_layers=n_layers, stride=stride, use_se=True)


class SE_PAResNet(PAResNetBase):

    def __init__(self, n_layers=50, stride=32, n_classes=1000):
        super(PAResNet, self).__init__()
        self.backbone = SE_PAResNetBackbone(n_layers, stride)
        self.classifier = nn.Linear(2048, n_classes, bias=True)



if __name__ == "__main__":
    #  layer1 = create_stage(64, 256, 3, 1, 1)
    #  layer2 = create_stage(256, 512, 4, 2, 1)
    #  layer3 = create_stage(512, 1024, 6, 1, 2)
    #  layer4 = create_stage(1024, 2048, 3, 1, 4)
    #  print(layer4)
    resnet = Resnet101()
    inten = torch.randn(1, 3, 224, 224)
    _, _, _, out = resnet(inten)
    print(out.size())
    for name, param in resnet.named_parameters():
        if 'bias' in name:
            print(name)


