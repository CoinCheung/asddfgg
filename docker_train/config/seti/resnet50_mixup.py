
n_gpus = 8
batchsize = 32
n_epoches = 20
n_eval_epoch = 1
#  opt_type = 'AdamW'
#  opt_args = dict(
#          #  lr=0.001 * (batchsize / 64) * n_gpus,
#          lr=0.001,
#          weight_decay=1e-2)
opt_type = 'SGD'
opt_args = dict(
        lr=0.1 * (batchsize / 128) * n_gpus,
        weight_decay=1e-4, nesterov=True, momentum=0.9)
schdlr_type = 'CosineLr'
schdlr_args = dict(
        max_iter=n_epoches, eta_ratio=0.,
        warmup_iter=10, warmup='linear', warmup_ratio=0.05)
model_args = dict(model_type='ResNet', in_chan=1, n_layers=50,
        n_classes=2, use_blur_pool=True, dropout=0.5,
        pretrain='/root/pretrained/not_uploaded/model_final_naive_r50_ra_lsr_200ep.pth')

dataset_args = dict(
        ds_type='SETI', root='./datasets/seti/',
        cropsize=320, binary=False
        )
print_freq = 50
metric = 'roc_auc'
num_workers = 4
grad_clip_norm = 10
ema_alpha = 0.999
use_mixed_precision = True
use_sync_bn = False
use_mixup = False
mixup_alpha = 1.0
use_cutmix = False
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
