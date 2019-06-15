import math
import torch


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_epochs,
            max_epochs,
            cos_eta=0,
            last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.cos_eta = cos_eta
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = self.last_epoch/self.warmup_epochs
            lr_group = [
                lr * warmup_factor for lr in self.base_lrs
            ]
        else:
            cos_last_epoch = self.last_epoch - self.warmup_epochs
            cos_epochs = self.max_epochs - self.warmup_epochs
            cos_factor = (1 + math.cos(math.pi * cos_last_epoch / cos_epochs)) / 2.
            lr_group = [
                lr * cos_factor for lr in self.base_lrs
            ]
        return lr_group
