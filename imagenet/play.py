import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
from apex import amp

from torch.nn import BatchNorm2d




class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn = BatchNorm2d(16)
        self.act = nn.ReLU(inplace=True)
        self.linear = nn.Linear(16, 1000)

    def forward(self, x):
        feat = self.act(self.bn(self.conv(x)))
        feat = torch.mean(feat, dim=(2, 3))
        logits = self.linear(feat)
        return logits

#
#  def main():
#      parse = argparse.ArgumentParser()
#      parse.add_argument('--local_rank',
#                         dest='local_rank',
#                         type=int,
#                         default=-1,)
#      args = parse.parse_args()
#
#      torch.cuda.set_device(args.local_rank)
#      dist.init_process_group(
#          backend='nccl',
#          init_method='env://',
#          world_size=torch.cuda.device_count(),
#          rank=args.local_rank
#      )
#
#      model = Model()
#      criteria = nn.CrossEntropyLoss()
#      model.cuda()
#
#      optimizer = torch.optim.SGD(
#          model.parameters(),
#          lr=0.016,
#          weight_decay=1e-5,
#          momentum=0.9
#      )
#
#      model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
#
#      local_rank = dist.get_rank()
#      model = nn.parallel.DistributedDataParallel(
#          model, device_ids=[local_rank, ], output_device=local_rank)
#
#      ims = torch.randn(1, 3, 224, 224).cuda()
#      lbs = torch.randint(0, 1000, (1, )).cuda()
#      logits = model(ims)
#      loss = criteria(logits, lbs)
#      loss.backward()
#      optimizer.step()
#      lr_scheduler.step()
#

def main():
    model = Model()
    criteria = nn.CrossEntropyLoss()
    model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.016,
        weight_decay=1e-5,
        momentum=0.9
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
    )

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    ims = torch.randn(1, 3, 224, 224).cuda()
    lbs = torch.randint(0, 1000, (1, )).cuda()
    logits = model(ims)
    loss = criteria(logits, lbs)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()


if __name__ == '__main__':
    main()

