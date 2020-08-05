
n_gpus = 8
batchsize = 128
n_epoches = 100
n_eval_epoch = 1
lr = 0.1 * (batchsize / 128) * n_gpus
opt_wd = 1e-4
nesterov = True
momentum = 0.9
warmup = 'linear'
warmup_ratio = 0.1
datapth = './imagenet/'
model_args = dict(model_type='resnet-50', n_classes=1000)
cropsize = 224
num_workers = 4
ema_alpha = 0.9999
use_mixed_precision = True
use_sync_bn = False
mixup = 'mixup'
mixup_alpha = 0.4
lb_smooth = 0.1
