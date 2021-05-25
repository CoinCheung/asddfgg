#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d
from .conv_ops import build_conv
from .blurpool import BlurPool




def build_act(act_type, chan):
    if act_type == 'relu':
        act = nn.ReLU(inplace=True)
    elif act_type == 'frelu':
        act = FReLU(chan)
    return act


class ConvBlock(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 ks=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_chan,
                           out_chan,
                           kernel_size=ks,
                           stride=stride,
                           padding=padding,
                           dilation=dilation,
                           groups=groups,
                           bias=False)
        self.norm = nn.BatchNorm2d(out_chan)
        self.act = nn.ReLU(inplace=True)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

    def fuse_conv_bn(self):
        self.conv = torch.nn.utils.fuse_conv_bn_eval(self.conv, self.norm)
        self.norm = nn.Identity()


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



class BasicBlock(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 stride=1,
                 stride_at_1x1=False,
                 dilation=1,
                 conv_type='nn',
                 act_type='relu',
                 use_se=False,
                 mid_type='nn',
                 use_blur_pool=False,
                 ):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBlock(in_chan, out_chan, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_chan, out_chan, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.proj = None
        if not in_chan == out_chan or not stride == 1:
            self.proj = nn.Sequential(
                    nn.Conv2d(in_chan, out_chan, 1, stride, 0),
                    nn.BatchNorm2d(out_chan)
                    )

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.bn2(feat)
        if not self.proj is None:
            x = self.proj(x)
        out = F.relu(x + feat, inplace=True)
        return out

    def fuse_conv_bn(self):
        pass



class Bottleneck(nn.Module):

    def __init__(self,
                 in_chan,
                 out_chan,
                 stride=1,
                 stride_at_1x1=False,
                 dilation=1,
                 conv_type='nn',
                 act_type='relu',
                 use_se=False,
                 mid_type='nn',
                 use_blur_pool=False,
                 ):
        super(Bottleneck, self).__init__()

        self.mid_type, self.conv_type = mid_type, conv_type
        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        assert not use_se
        mid_chan = out_chan // 4

        self.conv1 = build_conv(conv_type, in_chan,
                            mid_chan,
                            kernel_size=1,
                            stride=stride1x1,
                            bias=False)
        self.bn1 = nn.BatchNorm2d(mid_chan)
        self.relu1 = build_act(act_type, mid_chan)

        if use_blur_pool and stride3x3 == 2:
            stride3x3 = 1
            self.blur_pool = BlurPool(mid_chan, stride=2)
        else:
            self.blur_pool = nn.Identity()

        self.conv2 = build_conv(mid_type, mid_chan,
                            mid_chan,
                            kernel_size=3,
                            stride=stride3x3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        self.relu2 = build_act(act_type, mid_chan)
        self.conv3 = build_conv(conv_type, mid_chan,
                            out_chan,
                            kernel_size=1,
                            bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.bn3.last_bn = False if use_se else True

        self.use_se = use_se
        if use_se:
            self.se_att = SEBlock(out_chan, 16, conv_type=conv_type)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            skip_stride, self.skip_blur = stride, None
            if use_blur_pool and stride == 2:
                skip_stride = 1
                self.skip_blur = BlurPool(in_chan, stride=stride)
            self.downsample = nn.Sequential(
                build_conv(conv_type, in_chan, out_chan, kernel_size=1, stride=skip_stride, bias=False),
                nn.BatchNorm2d(out_chan)
            )


        self.relu3 = build_act(act_type, out_chan)


    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.blur_pool(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.use_se:
            residual = self.se_att(residual)

        inten = x
        if not self.downsample is None:
            if not self.skip_blur is None:
                x = self.skip_blur(x)
            inten = self.downsample(x)

        out = residual + inten

        #  out = self.relu(out)
        out = self.relu3(out)
        return out


    def fuse_conv_bn(self):
        if self.conv_type == 'nn':
            self.conv1 = torch.nn.utils.fuse_conv_bn_eval(
                    self.conv1, self.bn1)
            self.bn1 = nn.Identity()
        if self.mid_type == 'nn':
            self.conv2 = torch.nn.utils.fuse_conv_bn_eval(
                    self.conv2, self.bn2)
            self.bn2 = nn.Identity()
            self.conv3 = torch.nn.utils.fuse_conv_bn_eval(
                    self.conv3, self.bn3)
            self.bn3 = nn.Identity()
            if not self.downsample is None:
                self.downsample = torch.nn.utils.fuse_conv_bn_eval(
                    self.downsample[0], self.downsample[1])


class FuseModule(nn.Module):

    def __init__(self, n_chans=[48, 96, 192, 384],
            conv_type='nn', mid_type='nn', act_type='relu',
            use_blur_pool=False):
        super(FuseModule, self).__init__()
        self.n_branches = len(n_chans)
        fuse_layers = []
        for i in range(self.n_branches):
            layers = []
            for j in range(self.n_branches):
                if j < i: # downsample larger features
                    in_chan, out_chan = n_chans[j], n_chans[i]
                    conv_series = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_series.append(
                                nn.Sequential(
                                    nn.Conv2d(in_chan, out_chan, 3, 2, 1),
                                    nn.BatchNorm2d(out_chan))
                            )
                        else:
                            conv_series.append(
                                ConvBlock(in_chan, out_chan, 3, 2, 1)
                            )
                        in_chan = out_chan
                    layers.append(nn.Sequential(*conv_series))
                elif j == i: # do nothing
                    layers.append(nn.Identity())
                else: # upsample
                    # TODO: see if we can save resource by conv before interpolate
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2**(j - i), mode='bilinear',
                                align_corners=False),
                            nn.Conv2d(n_chans[j], n_chans[i], 1, 1, 0),
                            nn.BatchNorm2d(n_chans[i]))
                        )
            fuse_layers.append(nn.ModuleList(layers))
        self.fuse_layers = nn.ModuleList(fuse_layers)

    def forward(self, x):
        assert len(x) == self.n_branches
        outs = []
        for i in range(self.n_branches):
            feat = self.fuse_layers[i][0](x[0])
            for j in range(1, self.n_branches):
                feat = feat + self.fuse_layers[i][j](x[j])
            outs.append(feat)
        return outs



class HRModule(nn.Module):

    def __init__(self, n_branches=1, n_chans=[64,],
            block='bottleneck',
            use_se=False, conv_type='nn', mid_type='nn', act_type='relu',
            use_blur_pool=False):
        super(HRModule, self).__init__()
        assert n_branches == len(n_chans)
        self.n_branches = n_branches
        block = Bottleneck if block == 'bottleneck' else BasicBlock
        n_chans = [el * 4 for el in n_chans]
        self.branches = nn.ModuleList([
            self._make_one_branch(n_chans[ind], block)
            for ind in range(n_branches)])
        self.fuse_layer = FuseModule(n_chans)


    def forward(self, x):
        outs = [branch(inp) for branch, inp in zip(self.branches, x)]
        return self.fuse_layer(outs)

    def _make_one_branch(self, n_chan, block):
        # each branch has 4 blocks
        return nn.Sequential(
            block(n_chan, n_chan, stride=1),
            block(n_chan, n_chan, stride=1),
            block(n_chan, n_chan, stride=1),
            block(n_chan, n_chan, stride=1),
        )




class TransModule(nn.Module):

    def __init__(self, in_chans=[64,], out_chans=[48, 96],
            conv_type='nn', mid_type='nn', act_type='relu',
            use_blur_pool=False):
        super(TransModule, self).__init__()
        in_chans = [4 * el for el in in_chans]
        out_chans = [4 * el for el in out_chans]
        layers = []
        for ind, (in_chan, out_chan) in enumerate(zip(in_chans, out_chans)):
            if in_chan == out_chan:
                layers.append(nn.Identity())
            else:
                layers.append(ConvBlock(in_chan, in_chan, 3, 1, 1))
        self.layers = nn.ModuleList(layers)
        self.trans_down = ConvBlock(in_chan, in_chan, 3, 2, 1)
        self.n_branches = len(in_chans)

    def forward(self, x):
        outs = []
        for ind in range(self.n_branches):
            out = self.layers[ind](x[ind])
            outs.append(out)
        out = self.trans_down(x[-1])
        outs.append(out)
        return tuple(outs)



#         NUM_MODULES, NUM_BRANCHS, NUM_BLOCKS, NUM_CHANNELS, BLOCK
HRNET_PARAMS = {
    'hrnet_48': {
        'stage1': (1, 1, [4,], [64, ], 'bottleneck'),
        'stage2': (1, 2, [4, 4], [48, 96], 'basic'),
        'stage3': (4, 3, [4, 4, 4], [48, 96, 192], 'basic'),
        'stage4': (3, 4, [4, 4, 4, 4], [48, 96, 192, 384], 'basic'),
    },
    'hrnet_32': {
        'stage1': (1, 1, [4,], [64, ], 'bottleneck'),
        'stage2': (1, 2, [4, 4], [32, 64], 'basic'),
        'stage3': (4, 3, [4, 4, 4], [32, 64, 128], 'basic'),
        'stage4': (3, 4, [4, 4, 4, 4], [32, 64, 128, 256], 'basic'),
    },
    'hrnet_18': {
        'stage1': (1, 1, [4,], [64, ], 'bottleneck'),
        'stage2': (1, 2, [4, 4], [18, 36], 'basic'),
        'stage3': (4, 3, [4, 4, 4], [18, 36, 72], 'basic'),
        'stage4': (3, 4, [4, 4, 4, 4], [18, 36, 72, 144], 'basic'),
    },
}


class HRNetBackbone(nn.Module):

    def __init__(self, mtype='hrnet_18', in_chan=3, use_se=False,
            conv_type='nn', mid_type='nn', act_type='relu',
            use_blur_pool=False):
        super(HRNetBackbone, self).__init__()
        hparams = HRNET_PARAMS[mtype]
        # stem block
        self.conv1 = nn.Conv2d(in_chan, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)

        self.stage1 = nn.Sequential(
            Bottleneck(64, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
            Bottleneck(256, 256, stride=1),
        )
        self.trans1 = TransModule(
                in_chans=hparams['stage1'][3],
                out_chans=hparams['stage2'][3])

        self.stage2 = self.create_stage(hparams['stage2'])
        self.trans2 = TransModule(
                in_chans=hparams['stage2'][3],
                out_chans=hparams['stage3'][3])

        self.stage3 = self.create_stage(hparams['stage3'])
        self.trans3 = TransModule(
                in_chans=hparams['stage3'][3],
                out_chans=hparams['stage4'][3])

        self.stage4 = self.create_stage(hparams['stage4'])


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        feats1 = self.stage1(x)
        feats1 = self.trans1([feats1,])

        feats2 = self.stage2(feats1)
        feats2 = self.trans2(feats2)

        feats3 = self.stage3(feats2)
        feats3 = self.trans3(feats3)

        feats4 = self.stage4(feats3)
        return feats4

    def create_stage(self, s_hparams):
        return nn.Sequential(*[
                HRModule(s_hparams[1], s_hparams[3], s_hparams[4])
                for _ in range(s_hparams[0])])


    def load_ckpt(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=True)



class HRNet(nn.Module):

    def __init__(self, n_classes=1000, in_chan=3, n_layers=50, stride=32,
            use_se=False, conv_type='nn',
            mid_type='nn', act_type='relu',
            use_blur_pool=False):
        super(HRNet, self).__init__()
        self.backbone = HRNetBackbone()
        self.classifier = nn.Linear(
                self.backbone.out_chans[-1], n_classes, bias=True)

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

    def load_states(self, state):
        self.backbone.load_state_dict(state['backbone'])
        self.classifier.load_state_dict(state['classifier'])

    def fuse_conv_bn(self):
        for mod in self.modules():
            if mod == self: continue
            if hasattr(mod, 'fuse_conv_bn'): mod.fuse_conv_bn()
        return self



if __name__ == "__main__":
    net = HRNetBackbone()
    inten = torch.randn(1, 3, 224, 224)
    outs = net(inten)
    for o in outs:
        print(o.size())
    for name, param in resnet.named_parameters():
        if 'bias' in name:
            print(name)


