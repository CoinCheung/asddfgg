

import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBN(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        return feat

    def get_fused(self):
        return torch.nn.utils.fuse_conv_bn_eval(self.conv, self.bn)


class RepVGGModule(nn.Module):

    def __init__(self, in_chan, out_chan, stride=1, groups=1, dilation=1):
        super(RepVGGModule, self).__init__()
        self.identity = None
        if in_chan == out_chan and stride == 1:
            self.identity = nn.BatchNorm2d(in_chan)
        self.dense = ConvBN(in_chan, out_chan, ks=3, stride=stride, padding=dilation, groups=groups, dilation=dilation)
        #  pad_1x1 =  padding - ks // 2
        self.proj = ConvBN(in_chan, out_chan, ks=1, stride=stride, padding=0, groups=groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        feat_dense = self.dense(x)
        feat_proj = self.proj(x)
        feat_iden = 0
        if not self.identity is None:
            feat_iden = self.identity(x)
        feat = feat_dense + feat_proj + feat_iden
        feat = self.act(feat)
        return feat

    @torch.no_grad()
    def get_fused(self):
        dense = self.dense.get_fused()
        proj = self.proj.get_fused()

        # fuse bn and 1x1
        if not self.identity is None:
            bn_w, bn_b = self.identity.weight, self.identity.bias
            bn_rm, bn_rv = self.identity.running_mean, self.identity.running_var
            bn_eps = self.identity.eps
            bn_deno = torch.rsqrt(bn_rv + bn_eps)
            w = bn_w * bn_deno
            b = bn_b - bn_w * bn_rm * bn_deno

            proj.weight += torch.diag(w)[:, :, None, None]
            proj.bias += b
        # fuse 1x1 and 3x3
        dense.weight[:, :, 1, 1] += proj.weight[:, :, 0, 0]
        dense.bias += proj.bias
        return nn.Sequential(dense, self.act)


def create_state(in_chan, out_chan, n_blocks, stride, dilation=1):
    blocks = [RepVGGModule(in_chan, out_chan, stride, dilation=dilation), ]
    for i in range(1, n_blocks):
        blocks.append(RepVGGModule(out_chan, out_chan, 1, dilation=dilation))
    return nn.Sequential(*blocks)


PARAMS = {
        'a0': {
            'num_blocks': [2, 4, 14, 1],
            'width_exps': [0.75, 0.75, 0.75, 2.5]},
        'a1': {
            'num_blocks': [2, 4, 14, 1],
            'width_exps': [1., 1., 1., 2.5]},
        'a2': {
            'num_blocks': [2, 4, 14, 1],
            'width_exps': [1.5, 1.5, 1.5, 2.75]},
        'b0': {
            'num_blocks': [4, 6, 16, 1],
            'width_exps': [1., 1., 1., 2.5]},
        }


class RepVGGBackbone(nn.Module):

    def __init__(self, in_chan=3, mtype='a1', stride=32):
        super(RepVGGBackbone, self).__init__()
        width_exps = PARAMS[mtype]['width_exps']
        num_blocks = PARAMS[mtype]['num_blocks']

        stem_chan = min(64, int(64 * width_exps[0]))
        chans = [int((2 ** i) * 64 * width_exps[i]) for i in range(4)]
        dils = [1, 1] if stride == 32 else [el*(16//stride) for el in (1, 2)]
        strds = [2 if el == 1 else 1 for el in dils]
        dils = [1, 1] + dils
        strds = [2, 2] + strds

        self.stage0 = RepVGGModule(in_chan, stem_chan, stride=2, dilation=1)
        self.stage1 = create_state(stem_chan, chans[0],
                num_blocks[0], stride=strds[0], dilation=dils[0])
        self.stage2 = create_state(chans[0], chans[1],
                num_blocks[1], stride=strds[1], dilation=dils[1])
        self.stage3 = create_state(chans[1], chans[2],
                num_blocks[2], stride=strds[2], dilation=dils[2])
        self.stage4 = create_state(chans[2], chans[3],
                num_blocks[3], stride=strds[3], dilation=dils[3])
        self.out_chans = chans[-4:]

    def forward(self, x):
        feat2 = self.stage0(x)
        feat4 = self.stage1(feat2)
        feat8 = self.stage2(feat4)
        feat16 = self.stage3(feat8)
        feat32 = self.stage4(feat16)
        return feat4, feat8, feat16, feat32

    @torch.no_grad()
    def fuse_block(self):
        self.stage0 = self.stage0.get_fused()
        for name, child in self.named_children():
            if name == 'stage0': continue
            for ind, mod in enumerate(child):
                child[ind] = mod.get_fused()


class RepVGG(nn.Module):

    def __init__(self, in_chan=3, mtype='a1', n_classes=1000):
        super(RepVGG, self).__init__()
        self.backbone = RepVGGBackbone(in_chan=in_chan, mtype=mtype, stride=32)
        out_chan = self.backbone.out_chans[-1]
        self.classifier = nn.Linear(out_chan, n_classes)

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

    @torch.no_grad()
    def fuse_block(self):
        self.backbone.fuse_block()


if __name__ == "__main__":
    model = RepVGGBackbone(stride=16)
    model.eval()
    inten = torch.randn(2, 3, 224, 224)
    ind = -1
    out1 = model(inten)[ind]
    print(out1.size())
    #  print(model)
    model.fuse_block()
    #  print(model)
    out2 = model(inten)[ind]
    print(out2.size())

    print((out1 - out2).abs().max())
    #  print(model)

    #  inten = torch.randn(2, 4, 224, 224)
    #  model = RepVGGModule(4, 4, 3, 1, 1)
    #  model.eval()
    #  #  print(model)
    #  out1 = model(inten)
    #  model = model.get_fused()
    #  model.eval()
    #  #  print(model)
    #  out2 = model(inten)
    #  print(out2.sum())
    #  #  print(out1.sum())
    #  #  print(out2.sum())
    #  print((out1 - out2).abs().sum())
