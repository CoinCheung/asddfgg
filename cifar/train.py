
import os
import os.path as osp
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import Resnet18, WideResnet
from cifar import get_train_loader, get_val_loader
from ema import EMA
from lr_scheduler import WarmupCosineAnnealingLR, WarmupMultiStepLR, WarmupCyclicLR
from label_smooth import LabelSmoothSoftmaxCEV2, LabelSmoothSoftmaxCEV1
from loss import (
        KVDivLoss,
        SoftmaxCrossEntropyWithOneHot, OneHot, LabelSmooth
    )


##=========================
## settings
##=========================
# fix random behaviors
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
random.seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True

torch.multiprocessing.set_sharing_strategy('file_system')


##=========================
## configurations
##=========================
ds_name = 'cifar10'
n_classes = 10
pre_act = False
# dataloader
batchsize = 512
n_workers = 2
# optimizer
momentum = 0.9
wd = 5e-4
lr0 = 0.1 * (batchsize // 128)
lr_eta = 1e-5
n_warmup_epochs = 10
warmup_start_lr = 1e-5
warmup_method = 'linear'
n_epochs = 600
cycle_len = 190
cycle_mult = 1.
lr_decay = 1.
ema_alpha = 0.99
# label smooth
use_lb_smooth = False
lb_pos = 0.9
lb_neg = 0.00005
# mixup
use_mixup = False
mixup_use_kldiv = False
mixup_alpha = 0.3
# label refinery
lb_refine = False
refine_use_mixup = False
refine_mixup_alpha = 0.3
refine_cycles = 3


def set_model():
    model = WideResnet(n_classes, k=2, n=28) # wide resnet-28-2
    #  model = Resnet18(n_classes=n_classes, pre_act=pre_act)
    model.cuda()
    #  criteria = SoftmaxCrossEntropyWithOneHot()
    #  criteria = nn.BCEWithLogitsLoss()
    #  criteria = nn.CrossEntropyLoss()
    criteria = LabelSmoothSoftmaxCEV2(0.1)
    return model, criteria


def set_optimizer(model):
    optim = torch.optim.SGD(
        model.parameters(),
        lr=lr0,
        momentum=momentum,
        weight_decay=wd
    )
    #  lr_sheduler = WarmupMultiStepLR(
    #      optim,
    #      warmup_start_lr=warmup_start_lr,
    #      warmup_epochs=n_warmup_epochs,
    #      warmup=warmup_method,
    #      milestones=[60, 120, 160],
    #      gamma=0.2,
    #  )
    lr_sheduler = WarmupCosineAnnealingLR(
        optim,
        warmup_start_lr=warmup_start_lr,
        warmup_epochs=n_warmup_epochs,
        warmup=warmup_method,
        max_epochs=n_epochs,
        cos_eta=lr_eta,
    )
    #  lr_sheduler = WarmupCyclicLR(
    #      optim,
    #      warmup_start_lr=warmup_start_lr,
    #      warmup_epochs=n_warmup_epochs,
    #      warmup=warmup_method,
    #      max_epochs=n_epochs,
    #      cycle_len=cycle_len,
    #      cycle_mult=cycle_mult,
    #      lr_decay=lr_decay,
    #      cos_eta=lr_eta,
    #  )
    return optim, lr_sheduler


def train_one_epoch(
        model,
        criteria,
        dltrain,
        optim,
        ema,
        use_mixup=False,
        mixup_alpha=1,
    ):
    one_hot = OneHot(n_labels=n_classes)
    loss_epoch = []
    model.train()
    for _, (ims, lbs) in enumerate(dltrain):
        ims = ims.cuda()
        lbs = lbs.cuda()
        #  lbs = one_hot(lbs.cuda())

        logits = model(ims)
        loss = criteria(logits, lbs)

        #  if use_mixup:
        #      bs = ims.size(0)
        #      idx = torch.randperm(bs)
        #      lam = np.random.beta(mixup_alpha, mixup_alpha)
        #      ims_mix = lam * ims + (1.-lam) * ims[idx]
        #      lbs_mix = lam * lbs + (1.-lam) * lbs[idx]
        #      logits = model(ims_mix)
        #      loss = criteria(logits, lbs_mix)
        #      #  loss1 = criteria(logits, lbs)
        #      #  loss2 = criteria(logits, lbs[idx])
        #      #  loss = lam * loss1 + (1.-lam) * loss2
        #  else:
        #      logits = model(ims)
        #      if use_lb_smooth:
        #          lbs[lbs == 1] = lb_pos
        #          lbs[lbs == 0] = lb_neg
        #      loss = criteria(logits, lbs)
        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        loss_epoch.append(loss.item())
    loss_avg = sum(loss_epoch) / len(loss_epoch)
    return loss_avg


def save_model(model, save_pth):
    model.cpu()
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_pth)


def train(save_pth, use_mixup, mixup_alpha):
    model, criteria = set_model()
    ema = EMA(model, ema_alpha)

    optim, lr_sheduler = set_optimizer(model)

    dltrain = get_train_loader(
        batch_size=batchsize,
        num_workers=n_workers,
        dataset=ds_name,
        pin_memory=False
    )

    for e in range(n_epochs):
        tic = time.time()

        loss_avg = train_one_epoch(
            model, criteria, dltrain, optim, ema, use_mixup, mixup_alpha
        )
        lr_sheduler.step()
        acc = evaluate(model, verbose=False)
        ema.apply_shadow()
        acc_ema = evaluate(model, verbose=False)
        ema.restore()

        toc = time.time()
        msg = 'epoch: {}, loss: {:.4f}, lr: {:.4f}, acc: {:.4f}, acc_ema: {:.4f}, time: {:.2f}'.format(
            e,
            loss_avg,
            list(optim.param_groups)[0]['lr'],
            acc,
            acc_ema,
            toc - tic
        )
        print(msg)
    save_model(model, save_pth)
    print('done')
    return model



def evaluate(model, verbose=True):
    model.cuda()
    model.eval()

    batchsize = 500
    n_workers = 2
    dlval = get_val_loader(
        batch_size=batchsize,
        num_workers=n_workers,
        dataset=ds_name,
        pin_memory=True
    )

    matches = []
    all_preds = []
    for _, (ims, lbs) in enumerate(dlval):
        ims = ims.cuda()
        lbs = lbs.cuda()

        logits = model(ims)
        scores = F.softmax(logits, dim=1)
        _, preds = torch.max(scores, dim=1)
        match = lbs == preds
        matches.append(match)
        all_preds += preds.tolist()
    matches = torch.cat(matches, dim=0).float()
    acc = torch.mean(matches)
    if verbose:
        print('accuracy on test set is: {}/{} = {}'.format(
            torch.sum(matches),
            torch.numel(matches),
            acc
        ))
        id_cat = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        with open('./test_res.csv', 'w') as fw:
            fw.write('id,label\n')
            for i, el in enumerate(all_preds):
                fw.write('{},{}\n'.format(i+1, id_cat[el]))
    return acc


def main():
    savepth = './res/'
    save_name = 'model_final_naive.pth'
    if not osp.exists(savepth): os.makedirs(savepth)
    save_pth = osp.join(savepth, save_name)
    model = train(save_pth, use_mixup, mixup_alpha)
    evaluate(model, verbose=True)



if __name__ == "__main__":
    main()
