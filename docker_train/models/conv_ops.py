#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

import functools

#  from torch.nn import Conv2d as Conv2dWS
#  from torch.nn import Conv2d as SphereConv2d



class Conv2dACW(nn.Conv2d):

    def __init__(self, num_classes, class_ratio=None, bias=True):
        super(Conv2dACW, self).__init__(
            num_classes,
            num_classes,
            1,
            stride=1,
            padding=0,
            dilation=1,
            groups=num_classes,
            bias=bias)
        self.num_classes = num_classes
        self.class_ratio = class_ratio
        weight = torch.tensor(class_ratio).view(-1, 1, 1, 1)
        weight = torch.softmax(1. / weight, dim=1)
        self.weight.data.copy_(weight)

    def forward(self, x):
        weight = torch.softmax(self.weight, dim=1)
        return F.conv2d(x,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)


class OSLConv2d(nn.Conv2d):

    def __init__(self,
                 in_chan,
                 out_chan,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5
                ):
        super(OSLConv2d, self).__init__(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps
        self.mask = self.get_mask()

    def forward(self, x):
        weight = self.get_weight()
        return F.conv2d(x,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)

    def get_weight(self):
        N, _, _, _ = self.weight.size()
        #  weight = self.weight
        #  norm = weight.norm(2, dim=(1, 2, 3), keepdim=True)
        #  weight = weight / (norm + self.eps)
        weight = weight * self.mask
        return weight

    def get_mask(self):
        mask = torch.zeros_like(self.weight)
        o_chan, i_chan, _, _ = mask.size()
        assert i_chan >= o_chan
        bin_size = i_chan // o_chan
        for i in range(o_chan):
            mask[i, i * bin_size: (i + 1) * bin_size, :, :] = 1
        return mask.detach()

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if name == 'weight':
                    param = self.get_weight()
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data


class SphereConv2d(nn.Conv2d):

    def __init__(self,
                 in_chan,
                 out_chan,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5
                ):
        super(SphereConv2d, self).__init__(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        weight = self.get_weight()
        return F.conv2d(x,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)

    def get_weight(self):
        N, _, _, _ = self.weight.size()
        weight = self.weight
        norm = weight.norm(2, dim=(1, 2, 3), keepdim=True)
        weight = weight / (norm + self.eps)
        return weight

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if name == 'weight':
                    param = self.get_weight()
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data


class Conv2dWS(nn.Conv2d):

    def __init__(self,
                 in_chan,
                 out_chan,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5
                ):
        super(Conv2dWS, self).__init__(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.std_eps = eps

    def forward(self, x):
        weight = self.get_weight()
        return F.conv2d(x,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)

    def get_weight(self):
        N, _, _, _ = self.weight.size()
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - mean
        std = weight.std(dim=(1, 2, 3), keepdim=True) + self.std_eps
        weight = torch.div(weight, std)
        #  mean = weight.mean(
        #              dim=1,
        #              keepdim=True
        #         ).mean(
        #              dim=2,
        #              keepdim=True
        #         ).mean(
        #              dim=3,
        #              keepdim=True)
        #  weight = weight - mean
        #  std = weight.view(N, -1).std(dim=1).view(-1, 1, 1, 1) + self.std_eps
        #  weight = torch.div(weight, std)
        return weight

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if name == 'weight':
                    param = self.get_weight()
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data


class NormConv2d(nn.Conv2d):

    def __init__(self,
                 in_chan,
                 out_chan,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5
                ):
        super(NormConv2d, self).__init__(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        assert kernel_size == 1, 'does not support other kernel sizes'
        self.eps = eps

    def forward(self, x):
        N, _, _, _ = x.size()
        norm = x.norm(2, dim=1, keepdim=True)
        x = torch.div(x, norm + self.eps)
        weight = self.get_weight()
        return F.conv2d(x,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)

    def get_weight(self):
        N, _, _, _ = self.weight.size()
        weight = self.weight
        norm = weight.norm(2, dim=(1, 2, 3), keepdim=True)
        weight = weight / (norm + self.eps)
        return weight




## Weight Align
class Conv2dWA(nn.Conv2d):

    def __init__(self,
                 in_chan,
                 out_chan,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5
                ):
        super(Conv2dWA, self).__init__(
            in_chan,
            out_chan,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.std_eps = eps
        self.gamma = nn.Parameter(torch.randn(out_chan, 1, 1, 1))
        n = functools.reduce((lambda x, y: x * y), self.weight.size())
        self.const = (2 / n) ** 0.5

    def forward(self, x):
        weight = self.get_weight()
        return F.conv2d(x,
                        weight,
                        self.bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups)

    def get_weight(self):
        N, _, _, _ = self.weight.size()
        weight = self.weight
        mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight = weight - mean
        std = weight.std(dim=(1, 2, 3), keepdim=True)
        weight = torch.div(weight, std) * self.const * self.gamma + self.std_eps
        return weight

    def init_weight(self):
        nn.init.normal_(0, self.const)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        r"""Saves module state to `destination` dictionary, containing a state
        of the module, but not its descendants. This is called on every
        submodule in :meth:`~torch.nn.Module.state_dict`.

        In rare cases, subclasses can achieve class-specific behavior by
        overriding this method with custom logic.

        Arguments:
            destination (dict): a dict where state will be stored
            prefix (str): the prefix for parameters and buffers used in this
                module
        """
        for name, param in self._parameters.items():
            if param is not None:
                if name == 'weight':
                    param = self.get_weight()
                destination[prefix + name] = param if keep_vars else param.data
        for name, buf in self._buffers.items():
            if buf is not None:
                destination[prefix + name] = buf if keep_vars else buf.data


def build_conv(conv_type, in_chan, out_chan, kernel_size,
               stride=1, padding=0, dilation=1, groups=1,
               bias=True, **kwargs):
    if conv_type == 'nn':
        conv = nn.Conv2d(in_chan, out_chan, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias)
    if conv_type == 'wa':
        conv = Conv2dWA(in_chan, out_chan, kernel_size, stride=stride,
                padding=padding, dilation=dilation, groups=groups,
                bias=bias, **kwargs)
    return conv
