from bisect import bisect_right
import math
import torch


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_start_lr,
            warmup_epochs,
            max_epochs,
            warmup='exp',
            cos_eta=0,
            last_epoch=-1
    ):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup = warmup
        self.cos_eta = cos_eta
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # exp lr warmup
            if self.warmup == 'exp':
                lr_group = [
                    self.warmup_start_lr * (
                        pow(lr/self.warmup_start_lr, 1./self.warmup_epochs)
                        ** self.last_epoch
                    )
                    for lr in self.base_lrs
                ]
            # linear warmup
            elif self.warmup == 'linear':
                warmup_factor = self.last_epoch/self.warmup_epochs
                lr_group = [
                    self.warmup_start_lr
                    + (lr-self.warmup_start_lr) * warmup_factor
                    for lr in self.base_lrs
                ]
        else:
            cos_last_epoch = self.last_epoch - self.warmup_epochs
            cos_epochs = self.max_epochs - self.warmup_epochs
            cos_factor = (1 + math.cos(math.pi * cos_last_epoch / cos_epochs)) / 2.
            lr_group = [
                self.cos_eta + (lr - self.cos_eta) * cos_factor
                for lr in self.base_lrs
            ]
        return lr_group


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_start_lr,
            warmup_epochs,
            milestones,
            gamma=0.1,
            warmup='exp',
            last_epoch=-1
    ):
        self.warmup_start_lr = warmup_start_lr
        self.warmup_epochs = warmup_epochs
        self.milestones = milestones
        self.gamma = gamma
        self.warmup = warmup
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # exp lr warmup
            if self.warmup == 'exp':
                lr_group = [
                    self.warmup_start_lr * (
                        pow(lr/self.warmup_start_lr, 1./self.warmup_epochs)
                        ** self.last_epoch
                    )
                    for lr in self.base_lrs
                ]
            # linear warmup
            elif self.warmup == 'linear':
                warmup_factor = self.last_epoch/self.warmup_epochs
                lr_group = [
                    self.warmup_start_lr
                    + (lr-self.warmup_start_lr) * warmup_factor
                    for lr in self.base_lrs
                ]
        else:
            lr_group = [
                lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for lr in self.base_lrs
            ]
        return lr_group


class WarmupCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            warmup_start_lr,
            warmup_epochs,
            max_epochs,
            cycle_len,
            cycle_mult,
            lr_decay=1,
            warmup='exp',
            cos_eta=0,
            last_epoch=-1
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.cycle_len = cycle_len
        self.cycle_mult = cycle_mult
        self.lr_decay = lr_decay
        self.cos_eta = cos_eta
        self.warmup = warmup
        self.lr_decay_factor = 1
        self.n_cycles = 0
        self.curr_cycle_len = cycle_len
        self.cycle_past_all_epoch = 0
        super(WarmupCyclicLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # exp lr warmup
            if self.warmup == 'exp':
                lr_group = [
                    self.warmup_start_lr * (
                        pow(lr/self.warmup_start_lr, 1./self.warmup_epochs)
                        ** self.last_epoch
                    )
                    for lr in self.base_lrs
                ]
            # linear warmup
            elif self.warmup == 'linear':
                warmup_factor = self.last_epoch/self.warmup_epochs
                lr_group = [
                    self.warmup_start_lr
                    + (lr-self.warmup_start_lr) * warmup_factor
                    for lr in self.base_lrs
                ]
        else:
            cycle_epoch = (
                self.last_epoch - self.warmup_epochs - self.cycle_past_all_epoch
            )
            if cycle_epoch > self.curr_cycle_len:
                self.cycle_past_all_epoch += self.curr_cycle_len
                self.curr_cycle_len *= self.cycle_mult
                cycle_epoch = 0
                self.lr_decay_factor *= self.lr_decay
            cos_factor = 0.5 * (
                1 + math.cos(math.pi * cycle_epoch / self.curr_cycle_len)
            )
            lr_group = [
                self.cos_eta + (lr * self.lr_decay_factor - self.cos_eta)
                * cos_factor
                for lr in self.base_lrs
            ]
        return lr_group


if __name__ == '__main__':
    max_epochs = 210
    #  net = torch.nn.Conv2d(3, 16, 3, 1, 1)
    net = torch.nn.BatchNorm2d(16)
    op = torch.optim.SGD(net.parameters(), lr=1e-3)
    scheduler = WarmupCosineAnnealingLR(op, 1e-5, 10, max_epochs, 'linear')
    #  scheduler = WarmupCosineAnnealingLR(op, 10, max_epochs)
    #  scheduler = WarmupMultiStepLR(op, 1e-5, 10, [30, 60, 90], 0.1, 'linear')
    scheduler = WarmupCyclicLR(op, 1e-5, 10, 100, 90, 1, 0.8, 'linear', 1e-5)
    import numpy as np
    import matplotlib.pyplot as plt
    lrs = []
    for _ in range(max_epochs):
        lrs.append(scheduler.get_lr()[0])
        scheduler.step()
    xs = np.arange(len(lrs))
    lrs = np.array(lrs)
    plt.plot(xs, lrs)
    plt.grid()
    plt.show()

