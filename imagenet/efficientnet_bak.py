
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO :use swish rather than relu


def round_channels(n_chan, multiplier):
    new_chan = n_chan * multiplier
    new_chan = max(8, int(new_chan + 4) // 8 * 8)
    if new_chan < 0.9 * n_chan: new_chan += 8
    return new_chan


def act_func(x):
    return x * torch.sigmoid(x)
    # return F.relu(x, inplace=True)

def drop_connect(x, drop_ratio):
    keep_ratio = 1. - drop_ratio
    return x


class MBConv(nn.Module):

    def __init__(self, in_chan, out_chan, ks, stride=1, expand_ratio=1, se_ratio=0.25, skip=False):
        super(MBConv, self).__init__()
        assert ks in (3, 5, 7)
        self.expand = False
        expand_chan = in_chan
        if expand_ratio != 1:
            self.expand = True
            expand_chan = int(in_chan * expand_ratio)
            self.expand_conv = nn.Conv2d(
                in_chan,
                expand_chan,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.expand_bn = nn.BatchNorm2d(
                expand_chan, momentum=0.01, eps=1e-3)
        n_pad = (ks-1)//2
        self.dw_conv = nn.Conv2d(
            expand_chan,
            expand_chan,
            kernel_size=ks,
            padding=n_pad,
            groups=expand_chan,
            stride=stride,
            bias=False
        )
        self.dw_bn = nn.BatchNorm2d(expand_chan, momentum=0.01, eps=1e-3)
        self.use_se = False
        if se_ratio != 0:
            self.use_se = True
            se_chan = max(1, int(se_ratio*in_chan))
            self.se_conv1 = nn.Conv2d(expand_chan, se_chan, kernel_size=1)
            self.se_conv2 = nn.Conv2d(se_chan, expand_chan, kernel_size=1)

        self.proj_conv = nn.Conv2d(
            expand_chan,
            out_chan,
            kernel_size=1,
            bias=False
        )
        self.proj_bn = nn.BatchNorm2d(out_chan, momentum=0.01, eps=1e-3)
        self.skip = skip and (in_chan == out_chan) and (stride == 1)
        #  self.drop_connect_ratio = drop_connect_ratio

    def forward(self, x, drop_connect_ratio=0.2):
        feat = x
        if self.expand:
            feat = self.expand_conv(feat)
            feat = self.expand_bn(feat)
            feat = act_func(feat)
        feat = self.dw_conv(feat)
        feat = self.dw_bn(feat)
        feat = act_func(feat)
        if self.use_se:
            atten = torch.mean(feat, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
            atten = self.se_conv1(atten)
            atten = act_func(atten)
            atten = self.se_conv2(atten)
            atten = torch.sigmoid(atten)
            feat = feat * atten
        feat = self.proj_conv(feat)
        feat = self.proj_bn(feat)
        if self.skip:
            feat = drop_connect(feat, drop_connect_ratio)
            feat = feat + x
        return feat


class EfficientNetBackbone(nn.Module):

    def __init__(self, r_width=1., r_depth=1., dropout=0.2):
        super(EfficientNetBackbone, self).__init__()
        i_chans = [32, 16, 24, 40, 80, 112, 192]
        o_chans = [16, 24, 40, 80, 112, 192, 320]
        n_blocks = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expands = [1, 6, 6, 6, 6, 6, 6]
        drop_connect_ratios = []
        n_blocks = 0
        for i in range(7):
            in_chan = round_channels(i_chans[i], r_width)
            out_chan = round_channels(o_chans[i], r_width)
            repeats = int(math.ceil(r_depth * n_blocks[i]))
            for _ in range(repeats-1):
                drop_connect_ratio = 0.2 * float(idx) / len(self.blocks)

        out_chan_stem = round_channels(32, r_width)
        self.conv_stem = nn.Conv2d(
            3,
            out_chan_stem,
            kernel_size=3,
            padding=1,
            stride=2,
            bias=False,
        )
        self.bn_stem = nn.BatchNorm2d(out_chan_stem, momentum=0.01, eps=1e-3)

        layers = []
        blocks = []
        for i in range(7):
            if strides[i] != 1:
                layers.append(blocks)
                blocks = []
            in_chan = round_channels(i_chans[i], r_width)
            out_chan = round_channels(o_chans[i], r_width)
            repeats = int(math.ceil(r_depth * n_blocks[i]))
            blocks.append(MBConv(
                in_chan,
                out_chan,
                kernel_sizes[i],
                stride=strides[i],
                expand_ratio=expands[i],
                se_ratio=0.25,
                skip=True,))
                ## TODO: do not add drop connect here
                #  drop_connect_ratio=0.2))
            for _ in range(repeats-1):
                blocks.append(MBConv(
                    out_chan,
                    out_chan,
                    kernel_sizes[i],
                    stride=1,
                    expand_ratio=expands[i],
                    se_ratio=0.25,
                    skip=True,))
                    ## TODO: do not add drop connect here
                    #  drop_connect_ratio=0.2))
        layers.append(blocks)
        layers = [el for el in layers if any(el)]
        print(len(layers))
        self.layer1 = nn.ModuleList(layers[1])
        self.layer2 = nn.ModuleList(layers[2])
        self.layer3 = nn.ModuleList(layers[3])
        self.layer4 = nn.ModuleList(layers[4])
        self.layer5 = nn.ModuleList(layers[5])
        #  self.blocks = nn.ModuleList(blocks)

        head_chan = round_channels(1280, r_width)
        self.conv_head = nn.Conv2d(
            out_chan,
            head_chan,
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.bn_head = nn.BatchNorm2d(head_chan, momentum=0.01, eps=1e-3)

    def forward(self, x):
        feat = self.conv_stem(x)
        feat = self.bn_stem(feat)
        feat = act_func(feat)

        for idx, block in enumerate(self.blocks):
            drop_connect_ratio = 0.2 * float(idx) / len(self.blocks)
            feat = block(feat, drop_connect_ratio)
        feat = self.conv_head(feat)
        feat = self.bn_head(feat)
        feat = act_func(feat)
        return feat


if __name__ == '__main__':
    inten = torch.randn(16, 3, 224, 224).cuda()
    mbconv = MBConv(3, 32, 3, 1, 2, 0.25, True)
    mbconv.cuda()
    oten = mbconv(inten)
    loss = torch.mean(oten)
    loss.backward()
    print(oten.size())

    backbone = EfficientNetBackbone()
    backbone.cuda()
    feat =  backbone(inten)
    print(feat.size())
