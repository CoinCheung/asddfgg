
import time
import random
import torch
import torch.nn.functional as F
import numpy as np

from model import Resnet18
from cifar import get_train_loader, get_val_loader
from lr_scheduler import WarmupCosineAnnealingLR, WarmupMultiStepLR, WarmupCyclicLR
from label_smooth import LabelSmoothSoftmaxCE


# fix random behaviors
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
random.seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True

torch.multiprocessing.set_sharing_strategy('file_system')


## configurations
ds_name = 'cifar10'
n_classes = 10
pre_act = True
save_pth = './res/model_final_naive.pth'
mixup_alpha = 1.
use_mixup = False


def train():
    model = Resnet18(n_classes=n_classes, pre_act=pre_act)
    model.cuda()
    criteria = torch.nn.CrossEntropyLoss()
    #  criteria = LabelSmoothSoftmaxCE(
    #      lb_pos=0.95,
    #      lb_neg=0.00005,
    #  )

    batchsize = 256
    n_workers = 8
    dltrain = get_train_loader(
        batch_size=batchsize,
        num_workers=n_workers,
        dataset=ds_name,
        pin_memory=False
    )

    lr0 = 2e-1
    lr_eta = 1e-5
    momentum = 0.9
    wd = 5e-4
    n_epochs = 200
    n_warmup_epochs = 10
    warmup_start_lr = 1e-5
    warmup_method = 'linear'
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
    #      milestones=[100, 150],
    #      gamma=0.1,
    #  )
    #  lr_sheduler = WarmupCosineAnnealingLR(
    #      optim,
    #      warmup_start_lr=warmup_start_lr,
    #      warmup_epochs=n_warmup_epochs,
    #      warmup=warmup_method,
    #      max_epochs=n_epochs,
    #      cos_eta=lr_eta,
    #  )
    lr_sheduler = WarmupCyclicLR(
        optim,
        warmup_start_lr=warmup_start_lr,
        warmup_epochs=n_warmup_epochs,
        warmup=warmup_method,
        max_epochs=n_epochs,
        cycle_len=190,
        cycle_mult=1,
        cos_eta=lr_eta,
    )

    for e in range(n_epochs):
        tic = time.time()
        model.train()
        lr_sheduler.step()
        loss_epoch = []
        for _, (ims, lbs) in enumerate(dltrain):
            #  lr_sheduler.step()
            ims = ims.cuda()
            lbs = lbs.cuda()

            optim.zero_grad()
            if use_mixup:
                bs = ims.size(0)
                idx = torch.randperm(bs)
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                ims_mix = lam * ims + (1.-lam) * ims[idx]
                logits = model(ims_mix)
                loss1 = criteria(logits, lbs)
                loss2 = criteria(logits, lbs[idx])
                loss = lam * loss1 + (1.-lam) * loss2
            else:
                logits = model(ims)
                loss = criteria(logits, lbs)
            loss.backward()
            loss_epoch.append(loss.item())
            optim.step()
        model.eval()
        acc = evaluate(model, verbose=False)
        toc = time.time()
        msg = 'epoch: {}, loss: {:.4f}, lr: {:.4f}, acc: {:.4f}, time: {:.2f}'.format(
            e,
            sum(loss_epoch)/len(loss_epoch),
            list(optim.param_groups)[0]['lr'],
            acc,
            toc - tic
        )
        print(msg)

    model.cpu()
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, save_pth)



def evaluate(model=None, verbose=True):
    if model is None:
        model = Resnet18(n_classes=n_classes, pre_act=pre_act)
        state_dict = torch.load(save_pth)
        model.load_state_dict(state_dict)
        model.eval()
        model.cuda()

    batchsize = 100
    n_workers = 4
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



if __name__ == "__main__":
    train()
    evaluate()
