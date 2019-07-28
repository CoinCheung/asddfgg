#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxCrossEntropyWithOneHot(nn.Module):
    def __init__(self, reduction='mean',):
        super(SoftmaxCrossEntropyWithOneHot, self).__init__()
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        loss = logs * label

        if self.reduction == 'mean':
            loss = -torch.mean(torch.sum(loss, dim=1))
        elif self.reduction == 'none':
            loss = -torch.sum(loss, dim=1)
        return loss


class OneHot(nn.Module):
    def __init__(
            self,
            n_labels,
            lb_ignore=255,
        ):
        super(OneHot, self).__init__()
        self.n_labels = n_labels
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
        ignore = ignore.nonzero()
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(self.n_labels), *b]] = 0

        return lb_one_hot


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

