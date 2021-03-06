
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_ops import build_conv


class PS_Axial(nn.Module):

    def __init__(self, in_chan, q_chan=8, out_chan=16, kernel_size=3, padding=1, stride=1):
        super(PS_Axial, self).__init__()
        self.q_chan = q_chan
        self.ks = kernel_size
        self.pad = padding
        self.stride = stride
        self.qkv = nn.Sequential(
                nn.Conv2d(in_chan, q_chan * 4, 1, 1, 0),
                nn.BatchNorm2d(q_chan * 4))
        self.unfold = nn.Unfold(kernel_size, padding=padding)
        self.rq = nn.Parameter(torch.randn(q_chan, kernel_size * kernel_size), requires_grad=True)
        self.rk = nn.Parameter(torch.randn(q_chan, kernel_size * kernel_size), requires_grad=True)
        self.rv = nn.Parameter(torch.randn(q_chan * 2, kernel_size * kernel_size), requires_grad=True)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride)


    ##TODO: add bn
    def forward(self, x):
        n, c, h, w = x.size()
        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [self.q_chan, self.q_chan, 2 * self.q_chan], dim=1)
        q = q.view(n, self.q_chan, -1)
        k = k.view(n, self.q_chan, -1)
        v = v.view(n, self.q_chan * 2, -1)
        sc = torch.einsum('nci,ncj->nij', q, k)
        sc = sc + torch.einsum('ci,ncj->nij', self.rq, q)
        sc = sc + torch.einsum('ci,ncj->nij', self.rk, k)
        sc = sc.softmax(dim=2) # n(kk)(hw)
        v = v + self.rv.view(1, self.q_chan * 2, -1)
        out = torch.einsum('nij,ncj->nci', sc, v).view(n, -1, h, w)
        out = self.pool(out)
        return out

    #  def forward(self, x):
    #      n, c, h, w = x.size()
    #      #  h = (h - self.ks + self.pad) // self.stride
    #      #  w = (w - self.ks + self.pad) // self.stride
    #      qkv = self.qkv(x)
    #      q, k, v = torch.split(qkv, [self.q_chan, self.q_chan, 2 * self.q_chan], dim=1)
    #      # q -> nc(hw)
    #      # kv -> n(ckk)(hw)
    #      q = q.view(n, self.q_chan, -1)
    #      k = self.unfold(k).view(n, self.q_chan, -1, h * w)
    #      v = self.unfold(v).view(n, self.q_chan * 2, -1, h * w)
    #      sc = torch.einsum('ncj,ncij->nij', q, k)
    #      sc = sc + torch.einsum('ci,ncj->nij', self.rq, q)
    #      sc = sc + torch.einsum('ci,ncij->nij', self.rk, k)
    #      sc = sc.softmax(dim=1) # n(kk)(hw)
    #      v = v + self.rv.view(1, self.q_chan * 2, -1, 1)
    #      out = torch.einsum('nij,ncij->ncj', sc, v).view(n, -1, h, w)
    #      out = self.pool(out)
    #      return out


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
        self.conv = nn.Conv2d(in_chan,
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


class CABlock(nn.Module):

    def __init__(self, in_chan, out_chan, ratio=32):
        super(CABlock, self).__init__()
        mid_chan = max(8, in_chan // ratio)
        self.conv_cat = ConvBlock(in_chan, mid_chan, 1, 1, 0)
        self.conv_h = nn.Conv2d(mid_chan, out_chan, 1, 1, 0, bias=True)
        self.conv_w = nn.Conv2d(mid_chan, out_chan, 1, 1, 0, bias=True)

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = torch.mean(x, dim=(2,)).view(n, c, w, 1)
        x_w = torch.mean(x, dim=(3,)).view(n, c, h, 1)
        x_att = torch.cat([x_h, x_w], dim=2)
        x_att = self.conv_cat(x_att)

        x_att_h, x_att_w = x_att.split([h, w], dim=2)
        x_att_h = self.conv_h(x_att_h).sigmoid()
        x_att_w = self.conv_w(x_att_w).sigmoid()
        att = x_att_h.view(n, c, 1, w) * x_att_w
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


class GEAttentionModule(nn.Module):

    def __init__(self, in_chan, out_chan):
        super(GEAttentionModule, self).__init__()
        self.conv = ConvBlock(in_chan, out_chan, 3, 1, 1)
        self.conv_atten1 = ConvBlock(out_chan, out_chan, 3, 2, 1, groups=out_chan)
        self.conv_atten2 = ConvBlock(out_chan, out_chan, 3, 2, 1, groups=out_chan)
        self.conv_atten3 = nn.Sequential(
                    nn.Conv2d(out_chan, out_chan, 3, 2, 1,
                        groups=out_chan, bias=False),
                    nn.BatchNorm2d(out_chan)
                )
        self.sigmoid = nn.Sigmoid()
        nn.init.kaiming_normal_(self.conv_atten3.weight)
        nn.init.constant_(self.conv_atten3.bias, 0)

    def forward(self, x):
        feat = self.conv(x)
        size = feat.size()[2:]
        atten = self.conv_atten1(feat)
        atten = self.conv_atten2(atten)
        atten = self.conv_atten3(atten)
        atten = F.interpolate(atten, size, mode='bilinear', align_corners=False)
        atten = self.sigmoid(atten)
        out = torch.mul(feat, atten)
        return out



if __name__ == "__main__":
#  unfold = nn.Unfold(3, 1, 1)
    mod = PS_Axial(128, stride=2, kernel_size=56, padding=2)
    inten = torch.randn(4, 128, 56, 56)
#  out = unfold(inten)
    out = mod(inten)
    print(out.size())
