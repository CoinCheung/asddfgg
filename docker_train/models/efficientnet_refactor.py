
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_loss import SwishV3 as Activation
#  import nn.ReLU as Activation

## TODO :use swish rather than relu


from torch.nn import BatchNorm2d


def round_channels(n_chan, multiplier):
    new_chan = n_chan * multiplier
    new_chan = max(8, int(new_chan + 4) // 8 * 8)
    if new_chan < 0.9 * n_chan: new_chan += 8
    return new_chan


#  def act_func(x):
#      return x * torch.sigmoid(x)
#      #  return F.relu(x, inplace=True)


class DropConnect(nn.Module):

    def __init__(self, drop_ratio):
        super(DropConnect, self).__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if self.drop_ratio == 0 or (not self.training): return x
        batchsize = x.size(0)
        keep_ratio = 1. - self.drop_ratio
        with torch.no_grad():
            mask = torch.empty(batchsize, dtype=x.dtype, device=x.device)
            mask.bernoulli_(keep_ratio).div_(keep_ratio)
            mask = mask.detach()
        feat = x * mask.view(-1, 1, 1, 1)
        return feat


class ConvBNAct(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = BatchNorm2d(out_chan, momentum=0.01, eps=1e-3)
        self.act = Activation()

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.act(feat)
        return feat


class SEBlock(nn.Module):

    def __init__(self, in_chan, se_chan):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chan, se_chan, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(se_chan, in_chan, kernel_size=1, bias=True)
        self.act = Activation()

    def forward(self, x):
        atten = torch.mean(x, dim=(2, 3), keepdims=True)
        atten = self.conv1(atten)
        atten = self.act(atten)
        atten = self.conv2(atten)
        atten = torch.sigmoid(atten)
        return atten * x


class MBConv(nn.Module):

    def __init__(self, in_chan, out_chan, ks, stride=1, expand_ratio=1, se_ratio=0.25, skip=True, drop_connect_ratio=0.2):
        super(MBConv, self).__init__()
        assert ks in (3, 5, 7)

        # expand conv
        self.exp_conv = None
        exp_chan = int(in_chan * expand_ratio)
        if exp_chan != in_chan:
            self.exp_conv = ConvBNAct(in_chan, exp_chan, 1, 1, 0)
        # depthwise conv
        n_pad = (ks - 1) // 2
        self.dw_conv = ConvBNAct(exp_chan, exp_chan, ks, stride, n_pad, groups=exp_chan)
        # se-attention
        self.se_block = None
        if se_ratio != 0:
            se_chan = max(1, int(se_ratio * in_chan))
            self.se_block = SEBlock(exp_chan, se_chan)
        # project conv
        self.proj_conv = nn.Sequential(
            nn.Conv2d(exp_chan, out_chan, kernel_size=1, bias=False),
            BatchNorm2d(out_chan, momentum=0.01, eps=1e-3)
        )
        ## TODO: last_bn
        self.skip = skip and (in_chan == out_chan) and (stride == 1)
        self.drop_connect = DropConnect(drop_connect_ratio)

    def forward(self, x):
        feat = x
        if not self.exp_conv is None:
            feat = self.exp_conv(feat)
        feat = self.dw_conv(feat)
        if not self.se_block is None:
            feat = self.se_block(feat)
        feat = self.proj_conv(feat)

        if self.skip:
            feat = self.drop_connect(feat)
            feat = feat + x
        return feat


class EfficientNetStage(nn.Module):

    def __init__(self, in_chan, out_chan, ks, stride=1, expand_ratio=1, se_ratio=0.25, n_blocks=1, dc_ratios=[0, ]):
        super(EfficientNetStage, self).__init__()
        layers = []
        for i in range(n_blocks):
            b_stride = stride if i == 0 else 1
            b_in_chan = in_chan if i == 0 else out_chan
            layers.append(MBConv(b_in_chan, out_chan, ks, stride=b_stride,
                expand_ratio=expand_ratio, se_ratio=se_ratio,
                drop_connect_ratio=dc_ratios[i])
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


params_dict = {
       # width,depth,res,dropout
    'b0': (1.0, 1.0, 224, 0.2),
    'b1': (1.0, 1.1, 240, 0.2),
    'b2': (1.1, 1.2, 260, 0.3),
    'b3': (1.2, 1.4, 300, 0.3),
    'b4': (1.4, 1.8, 380, 0.4),
    'b5': (1.6, 2.2, 456, 0.4),
    'b6': (1.8, 2.6, 528, 0.5),
    'b7': (2.0, 3.1, 600, 0.5),
}

class EfficientNetBackbone(nn.Module):

    def __init__(self, model_type='b0'):
        super(EfficientNetBackbone, self).__init__()

        assert model_type in params_dict
        r_width, r_depth, _, _ = params_dict[model_type]

        out_chan_stem = round_channels(32, r_width)
        self.conv_stem = ConvBNAct(3, out_chan_stem, ks=3, padding=1, stride=2,)

        model_params = self.get_model_params(r_width, r_depth)
        for i, param in enumerate(zip(*model_params)):
            i_chan, o_chan, ks, strd, exp, n_b, dp_ratio = param
            self.add_module('layer{}'.format(i+1),
                EfficientNetStage(i_chan, o_chan, ks, strd,
                expand_ratio=exp, n_blocks=n_b, dc_ratios=dp_ratio)
            )
        self.out_chans = (model_params[1][1], model_params[1][2],
                model_params[1][4], model_params[1][6])


    def get_model_params(self, r_width=1., r_depth=1.):
        # b0 config
        i_chans = [32, 16, 24, 40, 80, 112, 192]
        o_chans = [16, 24, 40, 80, 112, 192, 320]
        n_blocks = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expands = [1, 6, 6, 6, 6, 6, 6]

        # expand
        i_chans = [round_channels(el, r_width) for el in i_chans]
        o_chans = [round_channels(el, r_width) for el in o_chans]
        n_blocks = [int(math.ceil(r_depth * el)) for el in n_blocks]

        # drop path ratios
        cum_n_blocks, n_all_blocks, dp_ratio = 0, sum(n_blocks), []
        for n_b in n_blocks:
            dp_ratio.append([
                0.2 * float(idx) / n_all_blocks
                for idx in range(cum_n_blocks, cum_n_blocks + n_b)
            ])
            cum_n_blocks += n_b
        return i_chans, o_chans, kernel_sizes, strides, expands, n_blocks, dp_ratio


    def forward(self, x):
        feat0 = self.conv_stem(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1) # feat4
        feat3 = self.layer3(feat2) # feat8
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4) # feat16
        feat6 = self.layer6(feat5)
        feat7 = self.layer7(feat6)
        return feat2, feat3, feat5, feat7


class EfficientNet(nn.Module):

    def __init__(self, model_type='b0', n_classes=1000):
        super(EfficientNet, self).__init__()
        assert model_type in params_dict
        r_width, _, _, r_dropout = params_dict[model_type]

        self.backbone = EfficientNetBackbone(model_type=model_type)
        o_chan = self.backbone.out_chans[-1]
        n_out_head_chan = round_channels(o_chan * 4, r_width)
        self.conv_out = ConvBNAct(o_chan, n_out_head_chan, ks=1, stride=1, padding=0)
        self.dropout = nn.Dropout(r_dropout) if r_dropout > 0.0 else None
        self.fc = nn.Linear(n_out_head_chan, n_classes, bias=True)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = self.conv_out(feat)
        feat = torch.mean(feat, dim=(2, 3))
        feat = self.dropout(feat) if not self.dropout is None else feat
        logits = self.fc(feat)
        return logits

    def get_states(self):
        state = {name: child.state_dict() for name, child in self.named_children()}
        #  state = dict(
        #      backbone=self.backbone.state_dict(),
        #      classifier=self.classifier.state_dict())
        return state



if __name__ == '__main__':
    inten = torch.randn(4, 3, 224, 224).cuda()
    mbconv = MBConv(3, 32, 3, 1, 2, 0.25, True)
    mbconv.cuda()
    oten = mbconv(inten)
    loss = torch.mean(oten)
    loss.backward()
    #  print(oten.size())

    backbone = EfficientNetBackbone()
    backbone.cuda()
    feat4, feat8, feat16, feat32 =  backbone(inten)
    print(feat4.size())
    print(feat8.size())
    print(feat16.size())
    print(feat32.size())
    #  #  print(backbone)
    #  print(backbone.n_chans)

    #  model = EfficientNet(model_type='b7', n_classes=1000)
    #  #  #  print(model)
    #  model.cuda()
    #  #  model.eval()
    #  logits = model(inten)
    #  print(logits.size())

    #  inten = torch.randn(4, 3, 5, 5).cuda()
    #  print(inten)
    #  dc = DropConnect(0.4)
    #  dc.cuda()
    #  out = dc(inten)
    #  print(out)
