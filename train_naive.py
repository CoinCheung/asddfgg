
import time
import random
import torch
import torch.nn.functional as F
import numpy as np

#  from model_new import Resnet18
from model import Resnet18
from cifar import get_train_loader, get_val_loader
from lr_scheduler import WarmupCosineAnnealingLR
from label_smooth import LabelSmoothSoftmaxCE

random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)

torch.multiprocessing.set_sharing_strategy('file_system')


## configurations
ds_name = 'cifar10'
n_classes = 10
pre_act = False
save_pth = './model_final_naive.pth'


def train():
    model = Resnet18(n_classes=n_classes, pre_act=pre_act)
    model.cuda()
    #  criteria = torch.nn.CrossEntropyLoss()
    criteria = LabelSmoothSoftmaxCE(
        lb_pos=0.9,
        lb_neg=0.0001,
    )

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
        cos_eta=lr_eta,
    )

    for e in range(n_epochs):
        tic = time.time()
        model.train()
        lr_sheduler.step()
        loss_epoch = []
        for _, (ims, lbs) in enumerate(dltrain):
            ims = ims.cuda()
            lbs = lbs.cuda()

            optim.zero_grad()
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
        #  all_preds = torch.tensor(all_preds).view(-1).cpu().tolist()
        id_cat = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        with open('./test_res.csv', 'w') as fw:
            fw.write('id,label\n')
            for i, el in enumerate(all_preds):
                fw.write('{},{}\n'.format(i+1, id_cat[el]))
    return acc



if __name__ == "__main__":
    train()
    evaluate()
