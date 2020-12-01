
n_gpus = 8
batchsize = 64
n_epoches = 120
n_eval_epoch = 1
opt_type = 'SGD'
opt_args = dict(
        lr=0.024 * (batchsize / 128) * n_gpus,
        weight_decay=1e-4, momentum=0.9)
schdlr_type = 'CosineLr'
schdlr_args = dict(
        max_iter=n_epoches, eta_ratio=0.,
        warmup_iter=5, warmup='linear', warmup_ratio=0.05)
grad_clip_norm = 10
model_args = dict(model_type='xception-41', n_classes=1000)
datapth = './imagenet/'
cropsize = 299
num_workers = 4
ema_alpha = 0.9999
use_mixed_precision = True
use_sync_bn = False
mixup = 'mixup'
mixup_alpha = 0.2
cutmix_beta = 1.
lb_smooth = 0.1

