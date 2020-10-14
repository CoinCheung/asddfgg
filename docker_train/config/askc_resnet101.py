
n_gpus = 8
batchsize = 128
n_epoches = 100
n_eval_epoch = 1
opt_type = 'SGD'
opt_args = dict(
        lr=0.1 * (batchsize / 128) * n_gpus,
        weight_decay=1e-4, nesterov=True, momentum=0.9)
schdlr_type = 'CosineLr'
schdlr_args = dict(
        max_iter=n_epoches, eta_ratio=0,
        warmup_iter=5, warmup='linear', warmup_ratio=0.05)
model_args = dict(model_type='askc-resnet-101', n_classes=1000)
grad_clip_norm = 10
datapth = './imagenet/'
cropsize = 224
num_workers = 4
ema_alpha = 0.9999
use_mixed_precision = True
use_sync_bn = False
use_mixup = True
mixup_alpha = 0.4
ues_cutmix = True
cutmix_beta = 1.
lb_smooth = 0.1


## org
#  n_gpus = 8
#  batchsize = 128
#  n_epoches = 100
#  #  n_epoches = 300
#  n_eval_epoch = 1
#  opt_type = 'SGD'
#  opt_args = dict(
#          lr=0.1 * (batchsize / 128) * n_gpus,
#          weight_decay=1e-4, nesterov=True, momentum=0.9)
#  #  schdlr_type = 'CosineLr'
#  #  schdlr_args = dict(
#  #          max_iter=n_epoches, eta_ratio=0,
#  #          warmup_iter=5, warmup='linear', warmup_ratio=0.05)
#  schdlr_type = 'StepLr'
#  schdlr_args = dict(
#          milestones=[30, 60, 90],
#          warmup_iter=0, warmup='linear', warmup_ratio=0.1)
#  grad_clip_norm = 10
#  datapth = './imagenet/'
#  model_args = dict(model_type='resnet-50', n_classes=1000)
#  cropsize = 224
#  num_workers = 4
#  ema_alpha = 0.9999
#  use_mixed_precision = True
#  use_sync_bn = False
#  use_mixup = False
#  mixup_alpha = 0.4
#  ues_cutmix = False
#  cutmix_beta = 1.
#  lb_smooth = 0.0
