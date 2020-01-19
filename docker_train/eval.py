import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader



def evaluate(ema, dl_eval):
    ema.apply_shadow()
    acc_1_ema, acc_5_ema = eval_model(ema.model, dl_eval)
    ema.restore()
    acc_1, acc_5 = eval_model(ema.model, dl_eval)
    torch.cuda.empty_cache()
    return acc_1, acc_5, acc_1_ema, acc_5_ema


@torch.no_grad()
def eval_model(model, dl_eval):
    model.eval()
    all_scores, all_gts = [], []
    for idx, (im, lb) in enumerate(dl_eval):
        im = im.cuda()
        lb = lb.cuda()
        logits = model(im)
        scores = torch.softmax(logits, dim=1)

        all_scores.append(scores)
        all_gts.append(lb)
    all_scores = torch.cat(all_scores, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_preds = torch.argsort(-all_scores, dim=1)
    match = (all_preds == all_gts.unsqueeze(1)).float()
    n_correct_1 = match[:, :1].sum()
    n_correct_5 = match[:, :5].sum()
    n_samples = torch.tensor(match.size(0)).cuda()
    dist.all_reduce(n_correct_1, dist.ReduceOp.SUM)
    dist.all_reduce(n_correct_5, dist.ReduceOp.SUM)
    dist.all_reduce(n_samples, dist.ReduceOp.SUM)
    acc1 = (n_correct_1 / n_samples).item()
    acc5 = (n_correct_5 / n_samples).item()
    torch.cuda.empty_cache()
    return acc1, acc5


if __name__ == "__main__":
    from common import Model, WeatherData
    model = Model()
    model.load_state_dict(torch.load('./model_final.pth'))
    model.cuda()
    model.eval()

    all_preds_t, all_preds_d, all_gts_t, all_gts_d = evaluate(model, 0)

    all_preds_t = torch.cat(all_preds_t, dim=0).cpu().numpy()
    all_preds_d = torch.cat(all_preds_d, dim=0).cpu().numpy()
    all_gts_t = torch.cat(all_gts_t, dim=0).cpu().numpy()
    all_gts_d = torch.cat(all_gts_d, dim=0).cpu().numpy()

    temperature_range = [-30, 50]
    dampness_range = [0, 100]

    all_preds_t = all_preds_t * (temperature_range[1] - temperature_range[0]) + temperature_range[0]
    all_preds_d = all_preds_d * (dampness_range[1] - dampness_range[0]) + dampness_range[0]
    all_gts_t = all_gts_t * (temperature_range[1] - temperature_range[0]) + temperature_range[0]
    all_gts_d = all_gts_d * (dampness_range[1] - dampness_range[0]) + dampness_range[0]

    import matplotlib
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(2, 1, sharex=True, sharey=False, num=1)
    x = np.arange(all_preds_t.shape[0])
    ax[0].plot(x, all_preds_t - all_gts_t, label='temperature')
    ax[0].set_title('temperature diff')
    ax[1].plot(x, all_preds_d - all_gts_d, label='dampness')
    ax[1].set_title('dampness diff')
    plt.savefig('res.jpg')
