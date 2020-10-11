#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d
#  from .conv_ops import Conv2dWS as Conv2d



class SEBlock(nn.Module):

    def __init__(self, in_chan, ratio, min_chan=8):
        super(SEBlock, self).__init__()
        mid_chan = in_chan // 8
        if mid_chan < min_chan: mid_chan = min_chan
        self.conv1 = nn.Conv2d(in_chan, mid_chan, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(mid_chan, in_chan, 1, 1, 0, bias=True)

    def forward(self, x):
        att = torch.mean(x, dim=(2, 3), keepdims=True)
        att = self.conv1(att)
        att = F.relu(att, inplace=True)
        att = self.conv2(att)
        att = torch.sigmoid(att)
        out = x * att
        return out



class Bottleneck(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 stride=1,
                 stride_at_1x1=False,
                 dilation=1,
                 use_se=False):
        super(Bottleneck, self).__init__()

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        mid_chan = out_chan // 4

        self.conv1 = Conv2d(in_chan,
                            mid_chan,
                            kernel_size=1,
                            stride=stride1x1,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.conv2 = Conv2d(mid_chan,
                            mid_chan,
                            kernel_size=3,
                            stride=stride3x3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.conv3 = Conv2d(mid_chan,
                            out_chan,
                            kernel_size=1,
                            bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        #  self.bn3.last_bn = True
        self.relu = nn.ReLU(inplace=True)

        self.use_se = use_se
        if use_se:
            self.se_att = SEBlock(out_chan, 16)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan)
            )
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = F.relu(residual, inplace=True)
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.use_se:
            residual = self.se_att(residual)

        inten = x
        if not self.downsample is None:
            inten = self.downsample(x)
        out = residual + inten
        out = self.relu(out)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, (Conv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(ly.weight)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


def create_stage(in_chan, out_chan, b_num, stride=1, dilation=1, use_se=False):
    assert out_chan % 4 == 0
    blocks = [Bottleneck(in_chan, out_chan, stride=stride, dilation=dilation, use_se=use_se),]
    for i in range(1, b_num):
        blocks.append(Bottleneck(out_chan, out_chan, stride=1, dilation=dilation, use_se=use_se))
    return nn.Sequential(*blocks)


class ResNetBackbone(nn.Module):

    def __init__(self, n_layers=50, stride=32, slim=True):
        super(ResNetBackbone, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el == 1 else 1 for el in dils]
        if n_layers == 50:
            layers = [3, 4, 6, 3]
        elif n_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError
        self.slim = slim

        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = Conv2d(3,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    ceil_mode=False)
        self.layer1 = create_stage(64, 256, layers[0], stride=1, dilation=1)
        self.layer2 = create_stage(256, 512, layers[1], stride=2, dilation=1)
        if self.slim:
            self.layer3 = create_stage(512, 512, layers[2], stride=strds[0], dilation=dils[0])
            self.layer4 = create_stage(512, 512, layers[3], stride=strds[1], dilation=dils[1])
        else:
            self.layer3 = create_stage(512, 1024, layers[2], stride=strds[0], dilation=dils[0])
            self.layer4 = create_stage(1024, 2048, layers[3], stride=strds[1], dilation=dils[1])

        self.init_weight()
        #  self.layers = []
        #  self.register_freeze_layers()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        ## init with msra method
        for name, md in self.named_modules():
            if isinstance(md, (Conv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(md.weight)
                if not md.bias is None: nn.init.constant_(md.bias, 0)

    def register_freeze_layers(self):
        self.layers = [self.conv1, self.bn1, self.layer1]

    def load_ckpt(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=True)

    def freeze_layers(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                param.requires_grad_(False)
            #  layer.eval()


class ResNet(nn.Module):

    def __init__(self, n_layers=50, stride=32, slim=False, n_classes=1000):
        super(ResNet, self).__init__()
        self.backbone = ResNetBackbone(n_layers, stride, slim)
        out_chan = 512 if slim else 2048
        self.classifier = nn.Linear(out_chan, n_classes, bias=True)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        logits = self.classifier(feat)
        return logits


class SEResNetBackbone(nn.Module):

    def __init__(self, n_layers=50, stride=32, slim=True):
        super(SEResNetBackbone, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el == 1 else 1 for el in dils]
        if n_layers == 50:
            layers = [3, 4, 6, 3]
        elif n_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError
        self.slim = slim

        self.bn0 = nn.BatchNorm2d(3)
        self.conv1 = Conv2d(3,
                            64,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    ceil_mode=False)
        self.layer1 = create_stage(64, 256, layers[0], stride=1, dilation=1, use_se=True)
        self.layer2 = create_stage(256, 512, layers[1], stride=2, dilation=1, use_se=True)
        if self.slim:
            self.layer3 = create_stage(512, 512, layers[2], stride=strds[0], dilation=dils[0], use_se=True)
            self.layer4 = create_stage(512, 512, layers[3], stride=strds[1], dilation=dils[1], use_se=True)
        else:
            self.layer3 = create_stage(512, 1024, layers[2], stride=strds[0], dilation=dils[0], use_se=True)
            self.layer4 = create_stage(1024, 2048, layers[3], stride=strds[1], dilation=dils[1], use_se=True)

        self.init_weight()
        #  self.layers = []
        #  self.register_freeze_layers()

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def init_weight(self):
        ## init with msra method
        for name, md in self.named_modules():
            if isinstance(md, (Conv2d, nn.Conv2d)):
                nn.init.kaiming_normal_(md.weight)
                if not md.bias is None: nn.init.constant_(md.bias, 0)

    def register_freeze_layers(self):
        self.layers = [self.conv1, self.bn1, self.layer1]

    def load_ckpt(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=True)

    def freeze_layers(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                param.requires_grad_(False)
            #  layer.eval()


class SEResNet(nn.Module):

    def __init__(self, n_layers=50, stride=32, slim=False, n_classes=1000):
        super(SEResNet, self).__init__()
        self.backbone = SEResNetBackbone(n_layers, stride, slim)
        out_chan = 512 if slim else 2048
        self.classifier = nn.Linear(out_chan, n_classes, bias=True)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        logits = self.classifier(feat)
        return logits


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


