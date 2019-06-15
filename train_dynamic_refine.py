
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Resnet18
from cifar import get_train_loader, get_val_loader
from lr_scheduler import WarmupCosineAnnealingLR

torch.manual_seed(123)

ds_name = 'cifar10'
n_classes = 10
pre_act = False


def train(gen_path, save_pth):
    model = Resnet18(n_classes=n_classes, pre_act=pre_act)
    model.train()
    model.cuda()
    criteria = nn.KLDivLoss(reduction='batchmean')
    generator = Resnet18(n_classes=10)
    state_dict = torch.load(gen_path)
    generator.load_state_dict(state_dict)
    generator.train()
    generator.cuda()

    batchsize = 256
    n_workers = 8
    dltrain = get_train_loader(
        batch_size=batchsize,
        num_workers=n_workers,
        dataset=ds_name,
        pin_memory=True
    )

    lr0 = 2e-1
    lr_eta = 1e-5
    momentum = 0.9
    wd = 5e-4
    n_epochs = 200
    n_warmup_epochs = 10
    optim = torch.optim.SGD(
        model.parameters(),
        lr=lr0,
        momentum=momentum,
        weight_decay=wd
    )
    lr_sheduler = WarmupCosineAnnealingLR(
        optim,
        warmup_epochs=n_warmup_epochs,
        max_epochs=n_epochs,
        cos_eta=lr_eta
    )

    for e in range(n_epochs):
        tic = time.time()
        model.train()
        lr_sheduler.step()
        loss_epoch = []
        for _, (ims, _) in enumerate(dltrain):
            ims = ims.cuda()
            with torch.no_grad():
                lbs = generator(ims).clone()
                lbs = torch.softmax(lbs, dim=1)
            optim.zero_grad()
            # generate labels
            logits = model(ims)
            probs = F.log_softmax(logits, dim=1)
            loss = criteria(probs, lbs)
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
    return model


def evaluate(model=None, save_pth=None, verbose=True):
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
    for _, (ims, lbs) in enumerate(dlval):
        ims = ims.cuda()
        lbs = lbs.cuda()

        logits = model(ims)
        scores = F.softmax(logits, dim=1)
        _, preds = torch.max(scores, dim=1)
        match = lbs == preds
        matches.append(match)
    matches = torch.cat(matches, dim=0).float()
    acc = torch.mean(matches)
    if verbose:
        print('accuracy on test set is: {}/{} = {}'.format(
            torch.sum(matches),
            torch.numel(matches),
            acc
        ))
    return acc



if __name__ == "__main__":
    gen_pths = [
        './res/model_final_naive.pth',
        './res/model_final_2.pth',
        './res/model_final_3.pth',
        './res/model_final_4.pth',
    ]
    save_pths = [
        './res/model_final_2.pth',
        './res/model_final_3.pth',
        './res/model_final_4.pth',
        './res/model_final_5.pth',
    ]

    for gen_pth, save_pth in zip(gen_pths, save_pths):
        model = train(gen_pth, save_pth)
        evaluate(model, save_pth, verbose=True)
