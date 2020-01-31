
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO :use swish rather than relu


from torch.nn import BatchNorm2d


def round_channels(n_chan, multiplier):
    new_chan = n_chan * multiplier
    new_chan = max(8, int(new_chan + 4) // 8 * 8)
    if new_chan < 0.9 * n_chan: new_chan += 8
    return new_chan


def act_func(x):
    #  return x * torch.sigmoid(x)
    return F.relu(x, inplace=True)


class DropConnect(nn.Module):

    def __init__(self, drop_ratio):
        super(DropConnect, self).__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if not self.training: return x
        batchsize = x.size(0)
        mask = torch.rand(batchsize).to(x.device)
        mask[mask<self.drop_ratio] = 0
        mask[mask>=self.drop_ratio] = 1
        feat = x * mask.view(-1, 1, 1, 1) / (1. - self.drop_ratio)
        return feat


class ConvBNAct(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = BatchNorm2d(out_chan, momentum=0.01, eps=1e-3)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = act_func(feat)
        return feat


class MBConv(nn.Module):

    def __init__(self, in_chan, out_chan, ks, stride=1, expand_ratio=1, se_ratio=0.25, skip=False, drop_connect_ratio=0.2):
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
            self.expand_bn = BatchNorm2d(
                expand_chan, momentum=0.01, eps=1e-3
            )
        n_pad = (ks - 1) // 2
        self.dw_conv = nn.Conv2d(
            expand_chan,
            expand_chan,
            kernel_size=ks,
            padding=n_pad,
            groups=expand_chan,
            stride=stride,
            bias=False
        )
        self.dw_bn = BatchNorm2d(expand_chan, momentum=0.01, eps=1e-3)
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
        self.proj_bn = BatchNorm2d(out_chan, momentum=0.01, eps=1e-3)
        self.skip = skip and (in_chan == out_chan) and (stride == 1)
        self.drop_connect = DropConnect(drop_connect_ratio)

    def forward(self, x):
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
        feat = act_func(feat)
        if self.skip:
            feat = self.drop_connect(feat)
            feat = feat + x
        return feat


class EfficientNetBackbone(nn.Module):

    def __init__(self, r_width=1., r_depth=1.):
        super(EfficientNetBackbone, self).__init__()

        layers, self.n_chans = [], []
        out_chan_stem = round_channels(32, r_width)
        self.n_chans.append(out_chan_stem)

        model_params = self.get_model_params(r_width, r_depth)
        for i in range(7):
            params = [el[i] for el in model_params]
            n_chan, blocks = 0, []
            for i_chan, o_chan, ks, stride, expand, drop_connect_ratio in zip(*params):
                blocks.append(MBConv(
                    i_chan,
                    o_chan,
                    ks,
                    stride=stride,
                    expand_ratio=expand,
                    drop_connect_ratio=drop_connect_ratio,
                    se_ratio=0.25,
                    skip=True,)
                )
                n_chan = o_chan
            layers.append(blocks)
            self.n_chans.append(n_chan)
        out_chan_head = round_channels(self.n_chans[-1] * 4, r_width)
        self.n_chans.append(out_chan_head)

        self.conv_stem = ConvBNAct(3, out_chan_stem, ks=3, padding=1, stride=2,)
        self.layer1 = nn.Sequential(*layers[0])
        self.layer2 = nn.Sequential(*layers[1])
        self.layer3 = nn.Sequential(*layers[2])
        self.layer4 = nn.Sequential(*layers[3])
        self.layer5 = nn.Sequential(*layers[4])
        self.layer6 = nn.Sequential(*layers[5])
        self.layer7 = nn.Sequential(*layers[6])
        self.conv_out = ConvBNAct(n_chan, out_chan_head, ks=1, stride=1, padding=0)


    def get_model_params(self, r_width=1., r_depth=1.):
        i_chans = [32, 16, 24, 40, 80, 112, 192]
        o_chans = [16, 24, 40, 80, 112, 192, 320]
        n_blocks = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        expands = [1, 6, 6, 6, 6, 6, 6]

        dec_i_chans = []
        dec_o_chans = []
        dec_kernel_sizes = []
        dec_strides = []
        dec_expands = []
        n_counts = []
        for i in range(7):
            count = 1
            dec_i_chans.append([])
            dec_o_chans.append([])
            dec_kernel_sizes.append([])
            dec_strides.append([])
            dec_expands.append([])
            in_chan = round_channels(i_chans[i], r_width)
            out_chan = round_channels(o_chans[i], r_width)
            repeats = int(math.ceil(r_depth * n_blocks[i]))
            dec_i_chans[i].append(in_chan)
            dec_o_chans[i].append(out_chan)
            dec_kernel_sizes[i].append(kernel_sizes[i])
            dec_strides[i].append(strides[i])
            dec_expands[i].append(expands[i])
            for _ in range(repeats-1):
                count += 1
                dec_i_chans[i].append(out_chan)
                dec_o_chans[i].append(out_chan)
                dec_kernel_sizes[i].append(kernel_sizes[i])
                dec_strides[i].append(1)
                dec_expands[i].append(expands[i])
            n_counts.append(count)
        all_counts = sum(n_counts)
        all_drop_connect_ratios = [
            0.2 * float(idx) / all_counts for idx in range(all_counts)
        ]
        dec_drop_connect_ratios = []
        for idx, n in enumerate(n_counts):
            dec_drop_connect_ratios.append([])
            for i in range(n):
                ratio = all_drop_connect_ratios.pop(0)
                dec_drop_connect_ratios[idx].append(ratio)
        return (dec_i_chans, dec_o_chans, dec_kernel_sizes, dec_strides,
                dec_expands, dec_drop_connect_ratios)

    def forward(self, x):
        feat0 = self.conv_stem(x)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1) # feat4
        feat3 = self.layer3(feat2) # feat8
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4) # feat16
        feat6 = self.layer6(feat5)
        feat7 = self.layer7(feat6)
        feat8 = self.conv_out(feat7) # feat32
        return feat0, feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8


class EfficientNet(nn.Module):
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

    def __init__(self, model_type='b0', n_classes=1000):
        super(EfficientNet, self).__init__()
        assert model_type in self.params_dict
        r_width, r_depth, _, r_dropout = self.params_dict[model_type]
        self.backbone = EfficientNetBackbone(r_width, r_depth)
        n_chans = self.backbone.n_chans
        self.dropout = nn.Dropout(r_dropout)
        self.fc = nn.Linear(n_chans[-1], n_classes, bias=True)

    def forward(self, x):
        feat = self.backbone(x)[-1]
        feat = torch.mean(feat, dim=(2, 3))
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits



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
    #  print(feat4.size())
    #  print(feat8.size())
    #  print(feat16.size())
    #  print(feat32.size())
    #  #  print(backbone)
    #  print(backbone.n_chans)

    model = EfficientNet(model_type='b7', n_classes=1000)
    #  print(model)
    model.cuda()
    model.eval()
    logits = model(inten)
    print(logits.size())

    #  inten = torch.randn(4, 3, 5, 5).cuda()
    #  print(inten)
    #  dc = DropConnect(0.4)
    #  dc.cuda()
    #  out = dc(inten)
    #  print(out)
