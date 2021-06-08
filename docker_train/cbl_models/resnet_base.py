#!/usr/bin/python
# -*- encoding: utf-8 -*-


import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d
from .conv_ops import build_conv
from .ibn import IBN
from .blurpool import BlurPool
from .att_block import ASKCFuse, CABlock, SEBlock, ConvBlock, GEAttentionModule

from pytorch_loss import FReLU



def init_weight(model):
    ## init with msra method
    for name, md in model.named_modules():
        if isinstance(md, (Conv2d, nn.Conv2d)):
            nn.init.kaiming_normal_(md.weight)
            if not md.bias is None: nn.init.constant_(md.bias, 0)


def build_act(act_type, chan):
    if act_type == 'relu':
        act = nn.ReLU(inplace=True)
    elif act_type == 'frelu':
        act = FReLU(chan)
    return act



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
                 use_ca=False,
                 ibn='none',
                 mid_type='nn',
                 use_blur_pool=False,
                 use_askc=False):
        super(Bottleneck, self).__init__()

        self.mid_type, self.conv_type, self.ibn = mid_type, conv_type, ibn
        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
        assert not (use_ca and use_se)
        mid_chan = out_chan // 4

        self.conv1 = build_conv(conv_type, in_chan,
                            mid_chan,
                            kernel_size=1,
                            stride=stride1x1,
                            bias=False)
        if ibn == 'a':
            self.bn1 = IBN(mid_chan)
        else:
            self.bn1 = nn.BatchNorm2d(mid_chan)
        #  self.relu1 = FReLU(mid_chan)
        self.relu1 = build_act(act_type, mid_chan)

        #  self.blur_pool = None
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
        #  self.relu2 = FReLU(mid_chan)
        self.relu2 = build_act(act_type, mid_chan)
        self.conv3 = build_conv(conv_type, mid_chan,
                            out_chan,
                            kernel_size=1,
                            bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.bn3.last_bn = False if (use_se or use_ca) else True
        #  self.relu = nn.ReLU(inplace=True)

        self.use_se, self.use_ca = use_se, use_ca
        if use_se:
            self.se_att = SEBlock(out_chan, 16, conv_type=conv_type)
        if use_ca:
            self.ca_att = CABlock(out_chan, out_chan, 32)

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

        self.use_askc = use_askc
        if use_askc:
            self.sc_fuse = ASKCFuse(out_chan, 4)

        self.inorm = None
        if ibn == 'b':
            self.inorm = nn.InstanceNorm2d(out_chan, affine=True)

        self.relu3 = build_act(act_type, out_chan)
        #  self.relu3 = FReLU(out_chan)


    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        #  residual = F.relu(residual, inplace=True)
        residual = self.relu1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.relu2(residual)
        residual = self.blur_pool(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.use_se:
            residual = self.se_att(residual)
        if self.use_ca:
            residual = self.ca_att(residual)

        inten = x
        if not self.downsample is None:
            if not self.skip_blur is None:
                x = self.skip_blur(x)
            inten = self.downsample(x)

        if self.use_askc:
            out = self.sc_fuse(residual, inten)
        else:
            out = residual + inten

        if not self.inorm is None:
            out = self.inorm(out)

        #  out = self.relu(out)
        out = self.relu3(out)
        return out


    def fuse_conv_bn(self):
        if self.conv_type == 'nn' and self.ibn == 'none':
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



def create_stage(in_chan, out_chan, b_num, stride=1, dilation=1,
            use_ca=False, use_se=False, use_askc=False, conv_type='nn',
            mid_type='nn', act_type='relu',
            ibn='none', use_blur_pool=False):
    assert out_chan % 4 == 0
    block_ibn = 'none' if ibn == 'b' else ibn
    blocks = [Bottleneck(in_chan, out_chan, stride=stride, dilation=dilation,
                use_ca=use_ca, use_se=use_se, use_askc=use_askc,
                conv_type=conv_type, mid_type=mid_type, act_type=act_type,
                ibn=block_ibn, use_blur_pool=use_blur_pool),]
    for i in range(1, b_num):
        block_ibn = 'none' if ibn == 'b' and i < b_num - 1 else ibn
        blocks.append(Bottleneck(out_chan, out_chan, stride=1,
                    dilation=dilation, use_ca=use_ca, use_se=use_se, use_askc=use_askc,
                    conv_type=conv_type, mid_type=mid_type, act_type=act_type,
                    ibn=block_ibn, use_blur_pool=use_blur_pool))
    return nn.Sequential(*blocks)



class ResNetBackbone(nn.Module):

    def __init__(self, in_chan=3, n_layers=50, stride=32, use_se=False,
            use_ca=False, use_askc=False, conv_type='nn', mid_type='nn', act_type='relu',
            ibn='none', stem_type='naive', use_ge_att=False, use_blur_pool=False, out1024=False):
        super(ResNetBackbone, self).__init__()
        self.mid_type = mid_type
        self.ibn = ibn
        self.act_type = act_type

        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [el*(16//stride) for el in (1, 2)]
        strds = [1, 2] + [2 if el == 1 else 1 for el in dils]
        if stem_type == 'res_d': strds[0] = 2

        if n_layers == 38:
            layers = [2, 3, 5, 2]
        elif n_layers == 50:
            layers = [3, 4, 6, 3]
        elif n_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError

        ## ibn
        ibns = ['none', 'none', 'none', 'none']
        if ibn == 'a':
            ibns = ['a', 'a', 'a', 'none']
        elif ibn == 'b':
            ibns = ['b', 'b', 'none', 'none']

        self.create_stem(in_chan, stem_type, conv_type, ibn,
                act_type, use_blur_pool)

        last_out_chan = 1024 if out1024 else 2048

        interv_block = GEAttentionModule if use_ge_att else nn.Identity


        self.layer1 = create_stage(64, 256, layers[0], stride=strds[0], dilation=1,
                    use_ca=use_ca, use_se=use_se, use_askc=use_askc, conv_type=conv_type,
                    mid_type=mid_type, act_type=act_type, ibn=ibns[0],
                    use_blur_pool=use_blur_pool)
        self.ge_att1 = interv_block(256, 256)
        self.layer2 = create_stage(256, 512, layers[1], stride=strds[1], dilation=1,
                    use_ca=use_ca, use_se=use_se, use_askc=use_askc, conv_type=conv_type,
                    mid_type=mid_type, act_type=act_type, ibn=ibns[1],
                    use_blur_pool=use_blur_pool)
        self.ge_att2 = interv_block(512, 512)
        self.layer3 = create_stage(512, 1024, layers[2], stride=strds[2],
                    dilation=dils[0], use_ca=use_ca, use_se=use_se, use_askc=use_askc,
                    mid_type=mid_type, conv_type=conv_type, act_type=act_type, ibn=ibns[2],
                    use_blur_pool=use_blur_pool)
        self.ge_att3 = interv_block(1024, 1024)
        self.layer4 = create_stage(1024, last_out_chan, layers[3], stride=strds[3],
                    dilation=dils[1], use_ca=use_ca, use_se=use_se, use_askc=use_askc,
                    mid_type=mid_type, conv_type=conv_type, act_type=act_type, ibn=ibns[3],
                    use_blur_pool=use_blur_pool)
        self.ge_att4 = interv_block(last_out_chan, last_out_chan)

        self.out_chans = [256, 512, 1024, last_out_chan]
        #  init_weight(self)
        self.layers = []
        #  self.register_freeze_layers()

    def create_stem(self, in_chan, stem_type, conv_type, ibn, act_type,
            use_blur_pool):
        if stem_type == 'naive':
            #  self.bn0 = nn.BatchNorm2d(3)
            conv_type0 = conv_type
            if conv_type == 'dy': conv_type0 = 'nn'
            self.conv1 = build_conv(conv_type0, in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if not use_blur_pool:
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                        dilation=1, ceil_mode=False)
            else:
                self.maxpool = nn.Sequential(
                        nn.MaxPool2d(kernel_size=2, stride=1),
                        BlurPool(64, stride=2))

        elif stem_type == 'res_d':
            self.conv1 = nn.Sequential(
                ConvBlock(in_chan, 32, 3, 2, 1),
                ConvBlock(32, 32, 3, 1, 1),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False))
            self.maxpool = nn.Identity()

        if self.ibn == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        #  relu = nn.ReLU(inplace=True)
        self.relu = build_act(act_type, chan=64)
        #  relu = FReLU(64)


    def forward(self, x):
        #  x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat4, feat8, feat16, feat32

    def register_freeze_layers(self):
        self.layers = [self.conv1, self.bn1, self.layer1]

    @torch.no_grad()
    def load_ckpt(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=True)

    def freeze_layers(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                param.requires_grad_(False)
            layer.eval()


class ResNet(nn.Module):

    def __init__(self, n_classes=1000, in_chan=3, n_layers=50, stride=32,
            use_se=False, use_ca=False, use_askc=False, conv_type='nn',
            mid_type='nn', act_type='relu', ibn='none', stem_type='naive',
            use_blur_pool=False, out1024=False, pretrain=None):
        super(ResNet, self).__init__()
        self.conv_type = conv_type
        self.backbone = ResNetBackbone(in_chan=in_chan, n_layers=n_layers,
                stride=stride, use_se=use_se, use_ca=use_ca, use_askc=use_askc,
                conv_type=conv_type, mid_type=mid_type, act_type=act_type,
                ibn=ibn, stem_type=stem_type, use_blur_pool=use_blur_pool,
                out1024=out1024)
        self.classifier = nn.Linear(self.backbone.out_chans[-1], n_classes, bias=True)
        self.load_pretrain(pretrain)

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

    @torch.no_grad()
    def load_states(self, state, strict=True):
        self.backbone.load_state_dict(state['backbone'])
        self.classifier.load_state_dict(state['classifier'])

    @torch.no_grad()
    def load_module_pretrain(self, module, state, prefix=''):
        logger = logging.getLogger()
        for k, v in module.state_dict().items():
            if not k in state:
                msg = f'skip {prefix}.{k} which is not in ckpt'
                logger.info(msg)
            elif not state[k].size() == v.size():
                s_size, v_size = state[k].size(), v.size()
                msg = f'skip {prefix}.{k} with size {v_size}, the ckpt size is {s_size}'
                logger.info(msg)
            else:
                v.copy_(state[k])

    @torch.no_grad()
    def load_pretrain(self, pth):
        if pth is None: return
        logger = logging.getLogger()
        logger.info(f'load pretrain from path: {pth}')
        state = torch.load(pth, map_location='cpu')
        self.load_module_pretrain(self.backbone, state['backbone'], 'backbone')
        self.load_module_pretrain(self.classifier, state['classifier'], 'classifier')


    def fuse_conv_bn(self):
        for mod in self.modules():
            if mod == self: continue
            if hasattr(mod, 'fuse_conv_bn'): mod.fuse_conv_bn()
        return self



### for denseCL pretrain
class ResNetDenseCL(nn.Module):

    def __init__(self, dim, n_classes=1000, in_chan=3, n_layers=50,
            stride=32,
            use_se=False, use_ca=False, use_askc=False, conv_type='nn',
            mid_type='nn', act_type='relu', ibn='none', stem_type='naive',
            use_blur_pool=False):
        super(ResNetDenseCLBase, self).__init__()
        self.backbone = ResNetBackbone(in_chan=in_chan, n_layers=n_layers,
                stride=stride, use_se=use_se, use_ca=use_ca, use_askc=use_askc,
                conv_type=conv_type, act_type=act_type, ibn=ibn,
                stem_type=stem_type, use_blur_pool=use_blur_pool)
        self.fc = nn.Linear(dim, n_classes, bias=True)
        self.dense_head = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim, n_classes, 1, 1, 0, bias=True))

    def forward(self, x):
        feat = self.backbone(x)[-1]
        x = torch.mean(feat, dim=(2, 3))
        logits = self.fc(x)
        dense = self.dense_head(feat)
        return logits, dense, feat

    def get_states(self):
        state = dict(
            backbone=self.backbone.state_dict(),
            fc=self.fc.state_dict(),
            dense_head=self.dense_head.state_dict(),
        )
        return state

    def load_states(self, state):
        self.backbone.load_state_dict(state['backbone'])
        self.fc.load_state_dict(state['fc'])
        self.dense_head.load_state_dict(state['dense_head'])


### for pixpro pretrain
class ResNetPixPro(nn.Module):

    def __init__(self, in_chan=3, n_layers=50, stride=32,
            use_se=False, use_ca=False, use_askc=False, conv_type='nn',
            mid_type='nn', act_type='relu', ibn='none', stem_type='naive',
            use_blur_pool=False):
        super(ResNetPixPro, self).__init__()
        self.backbone = ResNetBackbone(in_chan=in_chan, n_layers=n_layers,
                stride=stride, use_se=use_se, use_ca=use_ca, use_askc=use_askc,
                conv_type=conv_type, act_type=act_type, ibn=ibn,
                stem_type=stem_type, use_blur_pool=use_blur_pool)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        return feat

    def get_states(self):
        state = dict(
            backbone=self.backbone.state_dict(),
        )
        return state

    def load_states(self, state):
        self.backbone.load_state_dict(state['backbone'])

if __name__ == "__main__":
    #  layer1 = create_stage(64, 256, 3, 1, 1)
    #  layer2 = create_stage(256, 512, 4, 2, 1)
    #  layer3 = create_stage(512, 1024, 6, 1, 2)
    #  layer4 = create_stage(1024, 2048, 3, 1, 4)
    #  print(layer4)
    resnet = ResNetBase()
    inten = torch.randn(1, 3, 224, 224)
    out = resnet(inten)
    print(out.size())
    for name, param in resnet.named_parameters():
        if 'bias' in name:
            print(name)


