import torch
import torch.nn as nn
import torch.distributed as dist

from efficientnet import EfficientNet
from imagenet.imagenet import ImageNet


n_workers = 4
bs_per_gpu = 256


def evaluate(ema, dlval):
    org_states = ema.model.state_dict()
    ema.model.load_state_dict(ema.state)
    acc1, acc5 = eval_model(ema.model, dlval)
    ema.model.load_state_dict(org_states)
    return acc1, acc5


def eval_model(model, dlval):
    matches = []
    model.eval()
    with torch.no_grad():
        for _, (ims, lbs) in enumerate(dlval):
            ims = ims.cuda()
            lbs = lbs.cuda()

            logits = model(ims)
            scores = torch.softmax(logits, dim=1)
            rank_preds = torch.argsort(-scores, dim=1)
            match = (rank_preds == lbs.unsqueeze(1))
            matches.append(match)
    matches = torch.cat(matches, dim=0).float()
    n_correct_rank1 = matches[:, :1].float().sum().cuda()
    n_correct_rank5 = matches[:, :5].float().sum().cuda()
    n_samples = torch.tensor(matches.size(0)).float().cuda()
    print(n_correct_rank1)
    print(n_correct_rank5)
    print(n_samples)
    dist.all_reduce(n_correct_rank1, dist.ReduceOp.SUM)
    dist.all_reduce(n_correct_rank5, dist.ReduceOp.SUM)
    dist.all_reduce(n_samples, dist.ReduceOp.SUM)
    acc_rank1 = n_correct_rank1 / n_samples
    acc_rank5 = n_correct_rank5 / n_samples

    return acc_rank1.item(), acc_rank5.item()


def get_val_ds():
    dsval = ImageNet(root='./imagenet', mode='val', cropsize=cropsize)
    sampler_val = torch.utils.data.distributed.DistributedSampler(dsval)
    dlval = torch.utils.data.DataLoader(
        dsval,
        shuffle=False,
        sampler=sampler_val,
        batch_size=bs_per_gpu,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dlval


def main():
    dlval = get_val_ds()
    model = EfficientNet(model_type=model_type, n_classes=1000)


if __name__ == '__main__':
    main()
