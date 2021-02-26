#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv2d
from .conv_ops import build_conv
from .ibn import IBN

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
                           bias=bias)
        self.norm = nn.BatchNorm2d(out_chan)
        self.act = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

    def init_weight(self):
        nn.init.kaiming_normal_(self.conv.weight)
        if not self.conv.bias is None:
            nn.init.constant_(self.conv.bias, 0)

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


class LambdaLayer(nn.Module):
    def __init__(self, in_chan, out_chan, dim_k, n=None, r=None, heads=4, dim_u=1):
        super(LambdaLayer, self).__init__()

        assert (out_chan % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        dim_v = out_chan // heads

        self.to_q = nn.Conv2d(in_chan, dim_k * heads, 1, bias=False)
        self.to_k = nn.Conv2d(in_chan, dim_k * dim_u, 1, bias=False)
        self.to_v = nn.Conv2d(in_chan, dim_v * dim_u, 1, bias=False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = not r is None
        if self.local_contexts:
            assert (r % 2) == 1, 'Receptive kernel size should be odd'
            self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r),
                    padding=(0, r // 2, r // 2))
        else:
            assert (not n is None), 'You must specify the total sequence length (h x w)'
            self.pos_emb = nn.Parameter(torch.randn(n, n, dim_k, dim_u))

        self.heads = heads
        self.dim_u = dim_u # intra-depth dimension
        self.dim_k = dim_k
        self.dim_v = dim_v


    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.dim_u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = q.view(b, h, self.dim_k, -1)
        k = k.view(b, u, self.dim_k, -1)
        v = v.view(b, u, self.dim_v, -1)

        k = k.softmax(dim=-1)

        lam_c = torch.einsum('bukm,buvm->bkv', k, v)
        y_c = torch.einsum('bhkn,bkv->bnhv', q, lam_c)

        if self.local_contexts:
            v = v.view(b, u, -1, hh, ww)
            lam_p = self.pos_conv(v)
            y_p = torch.einsum('bhkn,bkvn->bnhv', q, lam_p.flatten(3))
        else:
            lam_p = torch.einsum('nmku,buvm->bnkv', self.pos_emb, v)
            y_p = torch.einsum('bhkn,bnkv->bnhv', q, lam_p)

        y = y_c + y_p
        out = y.view(b, hh, ww, -1).permute(0, 3, 1, 2)
        return out.contiguous()


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
                 ibn='none',
                 use_askc=False):
        super(Bottleneck, self).__init__()

        stride1x1, stride3x3 = (stride, 1) if stride_at_1x1 else (1, stride)
        assert out_chan % 4 == 0
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
        #  self.conv2 = Conv2d(mid_chan,
        self.conv2 = build_conv(conv_type, mid_chan,
                            mid_chan,
                            kernel_size=3,
                            stride=stride3x3,
                            padding=dilation,
                            dilation=dilation,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(mid_chan)
        #  self.relu2 = FReLU(mid_chan)
        self.relu2 = build_act(act_type, mid_chan)
        #  self.conv3 = Conv2d(mid_chan,
        self.conv3 = build_conv(conv_type, mid_chan,
                            out_chan,
                            kernel_size=1,
                            bias=False)
        self.bn3 = nn.BatchNorm2d(out_chan)
        self.bn3.last_bn = True if not use_se else False
        #  self.relu = nn.ReLU(inplace=True)

        self.use_se = use_se
        if use_se:
            self.se_att = SEBlock(out_chan, 16, conv_type=conv_type)

        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                #  Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                build_conv(conv_type, in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
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
        residual = self.conv3(residual)
        residual = self.bn3(residual)

        if self.use_se:
            residual = self.se_att(residual)

        inten = x
        if not self.downsample is None:
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


def create_stage(in_chan, out_chan, b_num, stride=1, dilation=1,
            use_se=False, use_askc=False, conv_type='nn', act_type='relu',
            ibn='none'):
    assert out_chan % 4 == 0
    block_ibn = 'none' if ibn == 'b' else ibn
    blocks = [Bottleneck(in_chan, out_chan, stride=stride, dilation=dilation,
                use_se=use_se, use_askc=use_askc,
                conv_type=conv_type, act_type=act_type, ibn=block_ibn),]
    for i in range(1, b_num):
        block_ibn = 'none' if ibn == 'b' and i < b_num - 1 else ibn
        blocks.append(Bottleneck(out_chan, out_chan, stride=1,
                    dilation=dilation, use_se=use_se, use_askc=use_askc,
                    conv_type=conv_type, act_type=act_type, ibn=block_ibn))
    return nn.Sequential(*blocks)


def create_stage_pa(in_chan, out_chan, b_num, stride=1, dilation=1,
            use_se=False, use_askc=False):
    assert out_chan % 4 == 0
    blocks = [PABottleneck(in_chan, out_chan, stride=stride, dilation=dilation,
                use_se=use_se, use_askc=use_askc),]
    for i in range(1, b_num):
        blocks.append(PABottleneck(out_chan, out_chan, stride=1,
                    dilation=dilation, use_se=use_se, use_askc=use_askc))
    return nn.Sequential(*blocks)



class ResNetBackboneBase(nn.Module):

    def __init__(self, n_layers=50, stride=32, use_se=False, use_askc=False, conv_type='nn', act_type='relu', ibn='none', res_type='naive'):
        super(ResNetBackboneBase, self).__init__()
        assert stride in (8, 16, 32)
        dils = [1, 1] if stride == 32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el == 1 else 1 for el in dils]
        if n_layers == 50:
            layers = [3, 4, 6, 3]
        elif n_layers == 101:
            layers = [3, 4, 23, 3]
        else:
            raise NotImplementedError

        self.res_type = res_type
        ## ibn
        ibns = ['none', 'none', 'none', 'none']
        if ibn == 'a':
            ibns = ['a', 'a', 'a', 'none']
        elif ibn == 'b':
            ibns = ['b', 'b', 'none', 'none']

        if self.res_type == 'naive':
            #  self.bn0 = nn.BatchNorm2d(3)
            conv_type0 = conv_type
            if conv_type == 'dy': conv_type0 = 'nn'
            self.conv1 = build_conv(conv_type0, 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if ibn == 'b':
                self.bn1 = nn.InstanceNorm2d(64, affine=True)
            else:
                self.bn1 = nn.BatchNorm2d(64)
            #  relu = nn.ReLU(inplace=True)
            self.relu = build_act(act_type, chan=64)
            #  relu = FReLU(64)
            #  self.conv1 = nn.Sequential(conv1, bn1, relu)
        elif self.res_type == 'res_d':
            self.conv1 = nn.Sequential(
                ConvBlock(3, 32, 3, 2, 1),
                ConvBlock(32, 32, 3, 1, 1),
                ConvBlock(32, 64, 3, 1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                dilation=1, ceil_mode=False)

        self.layer1 = create_stage(64, 256, layers[0], stride=1, dilation=1,
                    use_se=use_se, use_askc=use_askc, conv_type=conv_type,
                    act_type=act_type, ibn=ibns[0])
        self.layer2 = create_stage(256, 512, layers[1], stride=2, dilation=1,
                    use_se=use_se, use_askc=use_askc, conv_type=conv_type,
                    act_type=act_type, ibn=ibns[1])
        self.layer3 = create_stage(512, 1024, layers[2], stride=strds[0],
                    dilation=dils[0], use_se=use_se, use_askc=use_askc,
                    conv_type=conv_type, act_type=act_type, ibn=ibns[2])
        self.layer4 = create_stage(1024, 2048, layers[3], stride=strds[1],
                    dilation=dils[1], use_se=use_se, use_askc=use_askc,
                    conv_type=conv_type, act_type=act_type, ibn=ibns[3])

        #  init_weight(self)
        self.layers = []
        #  self.register_freeze_layers()


    def forward(self, x):
        #  x = self.bn0(x)
        x = self.conv1(x)
        if self.res_type == 'naive':
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

    def load_ckpt(self, path):
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state, strict=True)

    def freeze_layers(self):
        for layer in self.layers:
            for name, param in layer.named_parameters():
                param.requires_grad_(False)
            layer.eval()


class ResNetBase(nn.Module):

    def __init__(self):
        super(ResNetBase, self).__init__()

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



### for denseCL pretrain
class ResNetDenseCLBase(nn.Module):

    def __init__(self, dim, n_classes):
        super(ResNetDenseCLBase, self).__init__()
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
        self.fc.load_state_dict(state['classifier'])
        self.dense_head.load_state_dict(state['classifier'])



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


