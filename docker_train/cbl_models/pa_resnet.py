#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo

from torch.nn import Conv2d
from .conv_ops import build_conv
from .ibn import IBN


class SEBlock(nn.Module):

    def __init__(self, in_chan, ratio, min_chan=8, conv_type='nn'):
        super(SEBlock, self).__init__()
        mid_chan = in_chan // 8
        if mid_chan < min_chan: mid_chan = min_chan
        #  self.conv1 = nn.Conv2d(in_chan, mid_chan, 1, 1, 0, bias=True)
        #  self.conv2 = nn.Conv2d(mid_chan, in_chan, 1, 1, 0, bias=True)
        self.conv1 = build_conv(conv_type, in_chan, mid_chan, 1, 1, 0, bias=True)
        self.conv2 = build_conv(conv_type, mid_chan, in_chan, 1, 1, 0, bias=True)

    def forward(self, x):
        att = torch.mean(x, dim=(2, 3), keepdims=True)
        att = self.conv1(att)
        att = F.relu(att, inplace=True)
        att = self.conv2(att)
        att = torch.sigmoid(att)
        out = x * att
        return out


class ASKCFuse(nn.Module):

    def __init__(self, in_chan=64, ratio=16):
        super(ASKCFuse, self).__init__()
        mid_chan = in_chan // ratio
        self.local_att = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, 1, 1, 0, bias=True),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chan, in_chan, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_chan),)
        self.global_att = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan, 1, 1, 0, bias=True),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chan, in_chan, 1, 1, 0, bias=True),
            nn.BatchNorm2d(in_chan),)

    def forward(self, x1, x2):
        local_att = self.local_att(x1)
        global_att = torch.mean(x2, dim=(2, 3), keepdims=True)
        global_att = self.global_att(global_att)
        att = local_att + global_att
        att = torch.sigmoid(att)
        feat_fuse = x1 * att + x2 * (1. - att)
        return feat_fuse


class PABottleneck(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 stride=1,
                 stride_at_1x1=False,
                 dilation=1,
                 use_se=False,
                 use_askc=False):
        super(PABottleneck, self).__init__()

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        mid_chan = out_chan // 4

        self.bn1 = nn.BatchNorm2d(in_chan)
        self.conv1 = Conv2d(in_chan,
                            mid_chan,
                            kernel_size=1,
                            stride=stride1x1,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv2 = Conv2d(mid_chan,
                            mid_chan,
                            kernel_size=3,
                            stride=stride3x3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False)
        self.bn3 = nn.BatchNorm2d(mid_chan)
        self.conv3 = Conv2d(mid_chan,
                            out_chan,
                            kernel_size=1,
                            bias=True)

        self.use_se = use_se
        if use_se:
            self.se_att = SEBlock(out_chan, 16)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = Conv2d(in_chan, out_chan, kernel_size=1,
                    stride=stride, bias=True)

        self.use_askc = use_askc
        if use_askc:
            self.sc_fuse = ASKCFuse(out_chan, 4)


    def forward(self, x):
        inten = self.bn1(x)
        inten = F.relu(inten, inplace=True)
        residual = self.conv1(inten)
        residual = self.bn2(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.conv2(residual)
        residual = self.bn3(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.conv3(residual)

        if self.use_se:
            residual = self.se_att(residual)

        if not self.downsample is None:
            x = self.downsample(inten)

        if self.use_askc:
            out = self.sc_fuse(residual, x)
        else:
            out = residual + x

        #  if self.downsample is None:
        #      out = residual + x
        #  else:
        #      inten = self.downsample(inten)
        #      out = residual + inten
        return out



def create_stage_pa(in_chan, out_chan, b_num, stride=1, dilation=1,
            use_se=False, use_askc=False):
    assert out_chan % 4 == 0
    blocks = [PABottleneck(in_chan, out_chan, stride=stride, dilation=dilation,
                use_se=use_se, use_askc=use_askc),]
    for i in range(1, b_num):
        blocks.append(PABottleneck(out_chan, out_chan, stride=1,
                    dilation=dilation, use_se=use_se, use_askc=use_askc))
    return nn.Sequential(*blocks)



class PAResNetBackBoneBase(nn.Module):

    def __init__(self, n_layers=50, stride=32, use_se=False, use_askc=False):
        super(PAResNetBackBoneBase, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el == 1 else 1 for el in dils]
        if n_layers == 50:
            layers = [3, 4, 6, 3]
        elif n_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError

        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2,
                        padding=1, dilation=1, ceil_mode=False)
        self.layer1 = create_stage_pa(64, 256, layers[0], stride=1, dilation=1,
                    use_se=use_se,
                    use_askc=use_askc)
        self.layer2 = create_stage_pa(256, 512, layers[1], stride=2, dilation=1,
                    use_se=use_se,
                    use_askc=use_askc)
        self.layer3 = create_stage_pa(512, 1024, layers[2], stride=strds[0],
                    dilation=dils[0],
                    use_se=use_se,
                    use_askc=use_askc)
        self.layer4 = create_stage_pa(1024, 2048, layers[3], stride=strds[1],
                    dilation=dils[1],
                    use_se=use_se,
                    use_askc=use_askc)
        self.bn = nn.BatchNorm2d(2048)

        #  init_weight(self)
        #  self.layers = []
        #  self.register_freeze_layers()

    def forward(self, x):
        feat = self.bn0(x)
        feat = self.conv1(feat)
        feat = self.maxpool(feat)
        feat4 = self.layer1(feat)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        feat32 = self.bn(feat32)
        feat32 = F.relu(feat32, inplace=True)
        return feat4, feat8, feat16, feat32

    def register_freeze_layers(self):
        self.layers = [self.bn0, self.conv1, self.layer1]

    def load_ckpt(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=True)

    def freeze_layers(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                param.requires_grad_(False)
            layer.eval()


class PAResNetBase(nn.Module):

    def __init__(self):
        super(PAResNetBase, self).__init__()

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        logits = self.classifier(feat)
        return logits

    def get_states(self):
        state = dict(
            backbone=self.backbone.state_dict(),
            classifier=self.classifier.state_dict())
        return state

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

