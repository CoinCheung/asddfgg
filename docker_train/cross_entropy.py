#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp


class SoftmaxCrossEntropyV1(nn.Module):
    def __init__(self, reduction='mean',):
        super(SoftmaxCrossEntropyV1, self).__init__()
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        loss = logs * label

        if self.reduction == 'mean':
            loss = -torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'none':
            loss = -torch.sum(loss, dim=1)
        return loss


class CrossEntropyFunction(torch.autograd.Function):

    @staticmethod
    @amp.custom_fwd
    def forward(ctx, logits, label, reduction):
        loss = torch.log_softmax(logits, dim=1).neg_().mul_(label).sum(dim=1)

        ctx.logits = logits
        ctx.label = label
        ctx.reduction = reduction
        ctx.n_valid = loss.numel()

        if reduction == 'mean':
            loss = loss.mean()
        if reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):

        logits = ctx.logits
        label = ctx.label
        reduction = ctx.reduction
        n_valid = ctx.n_valid

        coeff = label.sum(dim=1, keepdim=True)

        scores = torch.softmax(logits, dim=1).mul_(coeff)
        if reduction == 'none':
            grad = scores.sub_(label).mul_(grad_output.unsqueeze(1))
        elif reduction == 'sum':
            grad = scores.sub_(label).mul_(grad_output)
        elif reduction == 'mean':
            grad = scores.sub_(label).mul_(grad_output.div_(n_valid))
        return grad, None, None


class SoftmaxCrossEntropyV2(nn.Module):

    def __init__(self, reduction='mean'):
        super(SoftmaxCrossEntropyV2, self).__init__()
        self.reduction = reduction

    def forward(self, logits, label):
        return CrossEntropyFunction.apply(
                logits, label, self.reduction)

#
#  class OnehotEncoder(nn.Module):
#      def __init__(
#              self,
#              n_classes,
#              lb_smooth=0,
#              ignore_idx=-1,
#          ):
#          super(OneHot, self).__init__()
#          self.n_classes = n_classes
#          self.lb_pos = 1. - lb_smooth
#          self.lb_neg = lb_smooth / n_classes
#          self.ignore_idx = ignore_idx
#
#      @ torch.no_grad()
#      def forward(self, label):
#          device = label.device
#          # compute output shape
#          size = list(label.size())
#          size.insert(1, self.n_classes)
#          if self.ignore_idx < 0:
#              out = torch.empty(size, device=device).fill_(
#                  self.lb_neg).scatter_(1, x.unsqueeze(1), self.lb_pos)
#          else:
#              # overcome ignore index
#              with torch.no_grad():
#                  label = label.clone().detach()
#                  ignore = label == self.ignore_idx
#                  label[ignore] = 0
#                  out = torch.empty(size, device=device).fill_(
#                      self.lb_neg).scatter_(1, x.unsqueeze(1), self.lb_pos)
#                  #  out = torch.empty(size, device=device).scatter_(
#                  #      self.lb_neg, x.unsqueeze(1), 1)
#                  ignore = ignore.nonzero()
#                  _, M = ignore.size()
#                  a, *b = ignore.chunk(M, dim=1)
#                  out[[a, torch.arange(self.n_classes), *b]] = 0
#          return out
#
#          #  N, *S = label.size()
#          #  size = [N, self.n_labels] + S
#          #  lb_one_hot = torch.zeros(size)
#          #  if label.is_cuda:
#          #      lb_one_hot = lb_one_hot.cuda()
#          #  ignore = label.data.cpu() == self.lb_ignore
#          #  label[ignore] = 0
#          #  lb_one_hot.scatter_(1, label.unsqueeze(1), 1)
#          #  ignore = ignore.nonzero()
#          #  _, M = ignore.size()
#          #  a, *b = ignore.chunk(M, dim=1)
#          #  lb_one_hot[[a, torch.arange(self.n_labels), *b]] = 0
#          #
#          #  return lb_one_hot
#
#
#  class OneHot(nn.Module):
#      def __init__(
#              self,
#              n_classes,
#              lb_smooth=0,
#              lb_ignore=255,
#          ):
#          super(OneHot, self).__init__()
#          self.n_labels = n_classes
#          self.lb_pos = 1 - lb_smooth
#          self.lb_neg = lb_smooth / n_classes
#          self.lb_ignore = lb_ignore
#
#      @ torch.no_grad()
#      def forward(self, label):
#          N, *S = label.size()
#          size = [N, self.n_labels] + S
#          lb_one_hot = torch.zeros(size)
#          if label.is_cuda:
#              lb_one_hot = lb_one_hot.cuda()
#          ignore = label.data.cpu() == self.lb_ignore
#          label[ignore] = 0
#          lb_one_hot.scatter_(1, label.unsqueeze(1), 1)
#          ignore = ignore.nonzero()
#          _, M = ignore.size()
#          a, *b = ignore.chunk(M, dim=1)
#          lb_one_hot[[a, torch.arange(self.n_labels), *b]] = 0
#
#          return lb_one_hot


class LabelSmooth(nn.Module):
    def __init__(
            self,
            n_labels,
            lb_pos=0.9,
            lb_neg=0.005,
            lb_ignore=255,
        ):
        super(LabelSmooth, self).__init__()
        self.n_labels = n_labels
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.lb_ignore = lb_ignore

    def forward(self, label):
        N, *S = label.size()
        size = [N, self.n_labels] + S
        lb_one_hot = torch.zeros(size)
        if label.is_cuda:
            lb_one_hot = lb_one_hot.cuda()
        ignore = label.data.cpu() == self.lb_ignore
        label[ignore] = 0
        lb_one_hot.scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0
        return label


class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(
            self,
            lb_pos=0.9,
            lb_neg=0.005,
            reduction='mean',
            lb_ignore=255,
        ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        label[[a, torch.arange(label.size(1)), *b]] = 0

        if self.reduction == 'mean':
            loss = -torch.sum(torch.sum(logs*label, dim=1)) / n_valid
        elif self.reduction == 'none':
            loss = -torch.sum(logs*label, dim=1)
        return loss


class KVDivLoss(nn.Module):
    def __init__(self, reduction='batchmean'):
        super(KVDivLoss, self).__init__()
        self.reduction = reduction
        self.kldivloss = nn.KLDivLoss(reduction=self.reduction)

    def forward(self, logits, labels):
        probs = F.log_softmax(logits, dim=1)
        loss = self.kldivloss(probs, labels)
        return loss


if __name__ == '__main__':
    torch.manual_seed(15)
    criteria = LabelSmoothSoftmaxCE(lb_pos=0.9, lb_neg=5e-3)
    net1 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(2, 3, 5, 5).cuda()
        lbs = torch.randint(0, 3, [2, 5, 5]).cuda()
        lbs[1, 3, 4] = 255
        lbs[1, 2, 3] = 255
        print(lbs)

    import torch.nn.functional as F
    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    #  loss1 = criteria1(logits1, lbs)
    loss = criteria(logits1, lbs)
    #  print(loss.detach().cpu())
    loss.backward()

    one_hot = OneHot(4)
    lb = torch.randint(0, 4, (2, 3, 3))
    lb[0, 0, 0] = 255
    print(lb)
    print(one_hot(lb))

