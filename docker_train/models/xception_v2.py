#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torch.nn import BatchNorm2d



BOTTLENECK_PARAMS = {
    'xception_41': {
        'entry_flow': (3, [2, 2, 2], [128, 256, 728]),
        'middle_flow': (8, 1, 728),
    },
    'xception_65': {
        'entry_flow': (3, [2, 2, 2], [128, 256, 728]),
        'middle_flow': (16, 1, 728),
    },
    'xception_71': {
        'entry_flow': (5, [2, 1, 2, 1, 2], [128, 256, 256, 728, 728]),
        'middle_flow': (16, 1, 728),
    },
}



class BNReLUConv(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, bias=False):
        super(BNReLUConv, self).__init__()
        self.bn = BatchNorm2d(in_chan)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        if not self.conv.bias is None: nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        x = self.conv(x)
        return x



class SepConvBlock(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
            dilation=1, with_act=False):
        super(SepConvBlock, self).__init__()

        self.with_act = with_act
        self.conv = nn.Conv2d(in_chan, in_chan, kernel_size=ks,
                stride=stride, padding=dilation,
                dilation=dilation, groups=in_chan, bias=False)
        self.bn1 = BatchNorm2d(in_chan)
        self.pairwise = nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False)
        self.bn2 = BatchNorm2d(out_chan)
        if with_act:
            self.act1 = nn.ReLU(inplace=True)
            self.act2 = nn.ReLU(inplace=True)

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        if self.with_act:
            x = self.act1(x)
        x = self.pairwise(x)
        x = self.bn2(x)
        if self.with_act:
            x = self.act2(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Block(nn.Module):

    def __init__(self,
            in_chans,
            out_chans,
            stride=1,
            dilation=1,
            sep_with_act=False,
            with_skip=True,
            ):
        super(Block, self).__init__()

        assert len(in_chans) == len(out_chans) == 3
        strides = [1, 1, stride]

        # residual
        self.residual = nn.ModuleList()
        for i in range(3):
            layer = [] if sep_with_act else [nn.ReLU(inplace=True)]
            layer.append(SepConvBlock(in_chans[i], out_chans[i],
                    ks=3, stride=strides[i],
                    padding=dilation, dilation=dilation,
                    with_act=sep_with_act))
            self.residual.append(nn.Sequential(*layer))

        # shortcut
        self.with_skip = with_skip
        skip_conv = False if stride == 1 and in_chans[0] == out_chans[-1] else True
        self.shortcut = None
        if with_skip and skip_conv:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_chans[0], out_chans[-1],
                        kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_chans[-1]))

        self.init_weight()

    def forward(self, x):
        feat1 = self.residual[0](x)
        feat2 = self.residual[1](feat1)
        feat3 = self.residual[2](feat2)

        if self.with_skip:
            sc = x
            if not self.shortcut is None:
                sc = self.shortcut(sc)
            feat3 = feat3 + sc
        return feat1, feat2, feat3

    def init_weight(self):
        if not self.shortcut is None:
            nn.init.kaiming_normal_(self.shortcut[0].weight, a=1)
            if not self.shortcut[0].bias is None:
                nn.init.constant_(self.shortcut[0].bias, 0)


class EntryFlow(nn.Module):

    def __init__(self, strides=[2, 2, 2], chans=[128, 256, 728]):
        super(EntryFlow, self).__init__()
        assert len(strides) == len(chans)
        in_chans, out_chans = self.compute_params(chans)
        self.aux_idx = [4, 7] if len(strides) == 3 else [7, 13]

        self.conv1 = BNReLUConv(3, 32, 3, 2, 1) # 1/2
        self.conv2 = BNReLUConv(32, 64, 3, 1, 1)

        self.layers = nn.ModuleList()
        for i, (s, i_ch, o_ch) in enumerate(zip(strides, in_chans, out_chans)):
            self.layers.append(Block(i_ch, o_ch, stride=s))
            #  self.add_module('block{}'.format(i), Block(i_ch, o_ch, stride=s))

    def forward(self, x):
        outs = []
        x = self.conv1(x)
        x = self.conv2(x)
        for child in self.layers:
            feats = child(x)
            x = feats[-1]
            outs += feats
        return x, tuple([outs[i] for i in self.aux_idx])

    def compute_params(self, chans):
        in_chans_ = [64, ] + [chans[i] for i in range(len(chans) - 1)]
        out_chans_ = chans
        in_chans, out_chans = [], []
        for ic, oc in zip(in_chans_, out_chans_):
            in_chans.append([ic, oc, oc])
            out_chans.append([oc, oc, oc])
        return in_chans, out_chans


class MiddleFlow(nn.Module):

    def __init__(self, n_blocks=8, n_chans=728, dilation=1):
        super(MiddleFlow, self).__init__()
        n_chans = [n_chans for _ in range(3)]
        for i in range(n_blocks):
            self.add_module('block{}'.format(i),
                    Block(n_chans, n_chans, dilation=dilation))

    def forward(self, x):
        for child in self.children():
            feats = child(x)
            x = feats[-1]
        return x


class ExitFlow(nn.Module):

    def __init__(self, strides=[2, 1], dilations=[1, 1]):
        super(ExitFlow, self).__init__()
        assert len(strides) == len(dilations) == 2

        in_chans, out_chans = [728, 728, 1024], [728, 1024, 1024]
        self.block1 = Block(in_chans, out_chans, stride=strides[0],
                dilation=dilations[0])

        in_chans, out_chans = [1024, 1536, 1536], [1536, 1536, 2048]
        self.block2 = Block(in_chans, out_chans, dilation=dilations[1],
            stride=strides[1], with_skip=False, sep_with_act=True)

    def forward(self, x):
        feats = self.block1(x)
        feats += self.block2(feats[-1])
        return feats[1], feats[-1]



class XceptionBackbone(nn.Module):

    def __init__(self, n_layers='41', stride=32):
        super(XceptionBackbone, self).__init__()
        assert stride in (8, 16, 32) and n_layers in ('41', '65', '71')
        params = BOTTLENECK_PARAMS['xception_{}'.format(n_layers)]
        ent_strides, ent_chans = params['entry_flow'][1], params['entry_flow'][2]
        n_mid_blocks, n_mid_chans = params['middle_flow'][0], params['middle_flow'][2]

        dilation_mid = 1
        ext_strides, ext_dilation = [2, 1], [1, 1]
        if stride == 16:
            ext_strides[0], dilation_ext = 1, [1, 2]
        elif stride == 8:
            ext_strides[0], dilation_ext = 1, [2, 4]
            ent_strides[-1], dilation_mid = 1, 2
        self.n_chans = (256, 728, 1024, 2048)

        self.entry_flow = EntryFlow(ent_strides, ent_chans)
        self.middle_flow = MiddleFlow(n_mid_blocks, n_mid_chans,
                dilation=dilation_mid)
        self.exit_flow = ExitFlow(strides=ext_strides,
                dilations=ext_dilation)

    def forward(self, x):
        feat_ent, feats_aux = self.entry_flow(x)
        feats_mid = self.middle_flow(feat_ent)
        feats_ext = self.exit_flow(feats_mid)
        outs = feats_aux + feats_ext
        return outs


class Xception(nn.Module):

    def __init__(self, n_layers, n_classes=1000):
        super(Xception, self).__init__()
        self.backbone = XceptionBackbone(n_layers=n_layers, stride=32)
        n_chan = self.backbone.n_chans[-1]
        self.classifier = nn.Linear(n_chan, n_classes)

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

class Xception41(Xception):

    def __init__(self, n_classes=1000):
        super(Xception41, self).__init__(n_layers='41')


class Xception65(Xception):

    def __init__(self, n_classes=1000):
        super(Xception65, self).__init__(n_layers='65')


class Xception71(Xception):

    def __init__(self, n_classes=1000):
        super(Xception71, self).__init__(n_layers='71')



if __name__ == "__main__":
    #  net = Xception71()
    #  net.train()
    #  net.cuda()
    #  Loss = nn.CrossEntropyLoss(ignore_index=255)
    #  import numpy as np
    #  inten = torch.tensor(np.random.randn(16, 3, 320, 240).astype(np.float32), requires_grad=False).cuda()
    #  label = torch.randint(0, 10, (16,)).cuda()
    #  for i in range(100):
    #      feat4, out = net(inten)
    #      logits = F.avg_pool2d(out, out.size()[2:]).view((16, -1))
    #      scores = F.softmax(logits, 1)
    #      loss = Loss(scores, label)
    #      loss.backward()
    #      print(i)
    #      print(out.size())

    import random
    import numpy as np
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.deterministic = True

    #  net1 = EntryFlow()
    #  net2 = MiddleFlow()
    #  net3 = ExitFlow()
    inten = torch.randn(2, 3, 224, 224)
    #  outs = net1(inten)
    #  for feat in outs:
    #      print(feat.size())
    #  out = net2(outs[-1])
    #  print(out.size())
    #  out = net3(out)
    #  print(out.size())

    print('==' * 5)
    net = XceptionBackbone(41, stride=32)
    outs = net(inten)
    for feat in outs:
        print(feat.size(), feat.sum().item())
    print(net.n_chans)

    #  net = Xception41(1000)
    net = Xception71(1000)
    logits = net(inten)
    print(logits.size())
