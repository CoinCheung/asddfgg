
import torch
import torch.nn as nn


class IBN(nn.Module):

    def __init__(self, in_chan, ratio):
        self.inorm_chan = int(in_chan * ratio)
        self.inorm = nn.InstanceNorm2d(self.inorm_chan, affine=True)
        self.bnorm = nn.BatchNorm2d(in_chan - self.inorm_chan)

    def forward(self, x):
        x_in, x_bn = torch.split(x, self.inorm_chan, dim=1)
        feat_in = self.inorm(x_in.contiguous())
        feat_bn = self.bnorm(x_bn.contiguous())
        feat = torch.cat([feat_in, feat_bn], dim=1)
        return feat
