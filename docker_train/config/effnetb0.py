
n_gpus = 8
batchsize = 128
n_epoches = 350
n_eval_epoch = 1
weight_decay = 1e-5
warmup = 'linear'
warmup_ratio = 0
opt_type = 'RMSpropTF'
opt_args = dict(
        lr=1.6e-2 * (batchsize / 256) * n_gpus,
        weight_decay=1e-5, momentum=0.9, alpha=0.9, eps=1e-3)
schdlr_type = 'ExpLr'
schdlr_args = dict(
        gamma=0.97, interval=2.4,
        warmup_iter=5, warmup='linear', warmup_ratio=0.)
grad_clip_norm = 10
datapth = './imagenet/'
model_args = dict(model_type='effcientnet-b0', n_classes=1000)
cropsize = 224
num_workers = 4
ema_alpha = 0.9999
use_mixed_precision = True
use_sync_bn = False
nesterov = True
mixup = 'mixup'
mixup_alpha = 0.2
lb_smooth = 0.1
