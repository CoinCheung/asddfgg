#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm2d


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = BatchNorm2d(out_chan)
        self.conv2 = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.00)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                BatchNorm2d(out_chan),
            )
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.dropout(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

    def init_weight(self):
        for _, md in self.named_modules():
            if isinstance(md, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not md.bias is None: nn.init.constant_(md.bias, 0)
        nn.init.constant_(self.bn2.weight, 0)  # gamma of last bn in residual path is initialized to be 0


class BottleneckBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BottleneckBlock, self).__init__()
        mid_chan = out_chan // 4
        self.conv1 = nn.Conv2d(
            in_chan,
            mid_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1 = BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(
            mid_chan,
            mid_chan,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(
            mid_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_chan,
                    out_chan,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                BatchNorm2d(out_chan),
            )
        self.init_weight()

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out

    def init_weight(self):
        for _, md in self.named_modules():
            if isinstance(md, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not md.bias is None: nn.init.constant_(md.bias, 0)
        nn.init.constant_(self.bn3.weight, 0)  # gamma of last bn in residual path is initialized to be 0


class BasicBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlockPreAct, self).__init__()
        self.bn1 = BatchNorm2d(in_chan)
        self.conv1 = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = BatchNorm2d(out_chan)
        #  self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(
            out_chan,
            out_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=False
            )
        self.init_weight()

    def forward(self, x):
        bn1 = self.bn1(x)
        act1 = self.relu(bn1)
        residual = self.conv1(act1)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        #  residual = self.dropout(residual)
        residual = self.conv2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(act1)

        out = shortcut + residual
        return out

    def init_weight(self):
        for _, md in self.named_modules():
            if isinstance(md, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not md.bias is None: nn.init.constant_(md.bias, 0)


class BottleneckBlockPreAct(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BottleneckBlockPreAct, self).__init__()
        mid_chan = out_chan // 4
        self.bn1 = BatchNorm2d(in_chan)
        self.conv1 = nn.Conv2d(
            in_chan,
            mid_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = BatchNorm2d(mid_chan)
        self.conv2 = nn.Conv2d(
            mid_chan,
            mid_chan,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn3 = BatchNorm2d(mid_chan)
        self.conv3 = nn.Conv2d(
            mid_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Conv2d(
                in_chan, out_chan, kernel_size=1, stride=stride, bias=False
            )
        self.init_weight()

    def forward(self, x):
        feat = self.bn1(x)
        feat = self.relu(feat)
        residual = self.conv1(feat)
        residual = self.bn2(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.bn3(residual)
        residual = self.relu(residual)
        residual = self.conv3(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(feat)

        out = shortcut + residual
        return out

    def init_weight(self):
        for _, md in self.named_modules():
            if isinstance(md, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    md.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not md.bias is None: nn.init.constant_(md.bias, 0)

##
## The following structures are designed for thumbnails such as cifar-10.
## If large images are used as dataset, the normal 5-stage model should be used.


class ResnetBackbone(nn.Module):
    def __init__(self, n_layers=20):
        super(ResnetBackbone, self).__init__()
        assert n_layers in (20, 32, 44, 56, 110)
        n_blocks = (n_layers - 2) // 6
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.create_layer(16, 16, bnum=n_blocks, stride=1)
        self.layer2 = self.create_layer(16, 32, bnum=n_blocks, stride=2)
        self.layer3 = self.create_layer(32, 64, bnum=n_blocks, stride=2)
        self.init_weight()

    def create_layer(self, in_chan, out_chan, bnum, stride=1):
        layers = [BasicBlock(in_chan, out_chan, stride=stride)]
        for _ in range(bnum-1):
            layers.append(BasicBlock(out_chan, out_chan, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat) # 1/2
        feat4 = self.layer3(feat2) # 1/4
        return feat2, feat4

    def init_weight(self):
        for _, child in self.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    child.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not child.bias is None: nn.init.constant_(child.bias, 0)


class ResnetPreActBackbone(nn.Module):
    def __init__(self, n_layers=20):
        super(ResnetPreActBackbone, self).__init__()
        assert n_layers in (20, 32, 44, 56, 110)
        n_blocks = (n_layers - 2) // 6
        self.bn1 = BatchNorm2d(3)
        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self.create_layer(16, 16, bnum=n_blocks, stride=1)
        self.layer2 = self.create_layer(16, 32, bnum=n_blocks, stride=2)
        self.layer3 = self.create_layer(32, 64, bnum=n_blocks, stride=2)
        self.bn_last = BatchNorm2d(64)
        self.relu_last = nn.ReLU(inplace=True)
        self.init_weight()

    def create_layer(self, in_chan, out_chan, bnum, stride=1):
        layers = [BasicBlockPreAct(in_chan, out_chan, stride=stride)]
        for _ in range(bnum-1):
            layers.append(BasicBlockPreAct(out_chan, out_chan, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        feat = self.bn1(x)
        feat = self.conv1(feat)

        feat = self.layer1(feat)
        feat2 = self.layer2(feat) # 1/2
        feat4 = self.layer3(feat2) # 1/4

        feat4 = self.bn_last(feat4)
        feat4 = self.relu_last(feat4)
        return feat2, feat4

    def init_weight(self):
        for _, child in self.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    child.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
                )
                if not child.bias is None: nn.init.constant_(child.bias, 0)


class Resnet18(nn.Module):
    def __init__(self, n_classes, pre_act=False):
        super(Resnet18, self).__init__()
        # according to paper, n_layers should be (20, 32, 44, 56, 110)
        n_layers = 20
        if pre_act:
            self.backbone = ResnetPreActBackbone(n_layers=n_layers)
        else:
            self.backbone = ResnetBackbone(n_layers=n_layers)
        #  self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(64, n_classes)
        self.bn = nn.BatchNorm1d(n_classes)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        #  feat = self.dropout(feat)
        feat = self.classifier(feat)
        feat = self.bn(feat)
        return feat

    def init_weight(self):
        nn.init.kaiming_normal_(
            self.classifier.weight, a=0, mode='fan_in', nonlinearity='leaky_relu'
        )
        if not self.classifier.bias is None:
            nn.init.constant_(self.classifier.bias, 0)

        #  for _, md in self.named_modules():
        #      if isinstance(md, BatchNorm2d):
        #          #  md.momentum = 1/(256**2)
        #          md.momentum = 0.1



if __name__ == "__main__":
    net = ResnetPreActBackbone(n_layers=20)
    x = torch.randn(2, 3, 224, 224)
    lb = torch.randint(0, 10, (2, )).long()
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    del net, out

    net = Resnet18(n_classes=10, pre_act=False)
    out = net(x)
    print(out.size())
    del net, out

    net = Resnet18(n_classes=10, pre_act=True)
    criteria = nn.CrossEntropyLoss()
    out = net(x)
    loss = criteria(out, lb)
    loss.backward()
    print(out.size())
