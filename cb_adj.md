目标
b0: 77.3(with autoaugment), 76.8(no autoaugment)
b1: 78.8/94.4

baseline: 
69.02/89.11

加上scheduler
68.89/88.96

label_smooth: 
68.54/88.79

拿到611上，并且修正crop_eval:
68.14/88.42

下面是重启的docker, 换lb-smoothv1: 
apex-fp16: 
    lb-smoothv2:  loss inf
    lb-smmothv2,  log_softmax:  67.86/88.54 
    lb-smoothv1: 


在141上做的比较apex和master: 
原版master: 141-screen 
使用apex的fp32: 68.15/88.59
使用apex的fp16: 68.27/88.78
使用apex的fp16, bs变2倍: 

fp16 + lb-smmothv2 + 1024:  67.86/88.54 
fp16 + lb-smmothv2 + 2048:  68.2/88.54 

fp16 + lb-smooothv2 + 1024 + wd=1e-5: 70.75/89.83
fp16 + lb-smooothv2 + 1024 + wd=2e-5: 71.03/90.13


看rmsprob哪里的问题。 rmsprop:
    wd=1e-5:
    wd=2e-5 + 200 epoch: 1/4
    不加wd: 爆了nan
    warmup从0开始exp: + 100 epoch: 0.1/0.5
    warmup从0开始linear: screen: 爆了nan
    下面是warmup + liner: 
        init + wd=0: 正常
        wd = 0:  正常
        init:  30个epoch, 还没收敛
        init + wd = l2: 100epoch不收敛
        wd = l2: 不看了
    fp32:
    master: 
    lr改小: 

    fp16 + init + wd=0: scale=0了, loss=nan
    fp32 + init + wd=0:  30.58/57.71
    fp16 + init + wd=直接decay: 
    目前的结论:
        只要有wd, rmsprop就不收敛, 不管是optimizer里面的, 还是单独加上的

141: 旧的lb-smooth: 71.02/90.05
     修正lb-smooth: 71.13/90.16
     原始的ce: 70.73/89.83
     重启了docker: 
        改进的lb-smooth:70.84/89.82 
        只使用l2(1e-5): 69.58/89.04
        只使用wd(1e-5) + 最后的dist.barrier(): 69.69/89.16
611 + fp16:
        不给bn和bias加wd 1xbs: 69.36/88.94
        不给bn和bias加wd 2xbs: 69.3/88.96
        mbconv的proj-conv后面加上act: 68.46/88.4
        顺序改成model-optim-apex-ema-syncbn-ddp-schdlr:68.43/88.46
        加上cuda.sychronize()看两个epoch之后一样不一样, 是否还有warning:
614 + fp16:
        rmsprobtf: 73.68/91.53
        改成450个epoch: 73.86/91.57
        加上color-jitter(0.4)和random-erasing(0.2): 73.37/91.45 --  应该是没收敛完
        学习率以128为base来算: 73.85/91.59
        warmup也算在real-iter里面: 
        warmup从1e-6开始: 
        warmup使用linear:
    proj_bn的gamma改成0: 


611: 官方efficientnet + swish: 75.05/92.27
611: 官方efficientnet + relu: 74.42/91.91
611: 官方efficientnet + relu + 官方 scheduler: 
614: 官方efficientnet + relu: 74.35/91.81
614: 官方efficientnet + swish + 官方rand_aug: 75.16/92.43
614: 官方efficientnet + swish + 官方rand_aug + 官方scheduler: 75.3/92.29
614: 官方efficientnet + swish + 官方rand_aug + official_init: 75.21/92.5
614: 官方efficientnet + swish + cv2的transform/randomaug + official_init: 74.64/92.15

614: 官方efficientnet + swish + cv2的transform不加randaug + official_init: 74.78/91.99
614: 官方efficientnet + swish + 官方的transform不加randaug + official_init: 74.28/91.91
614: 官方efficientnet + swish + cv2的transform加自制randaug10 + official_init: 74.74/91.96
614: 官方efficientnet + swish + cv2的transform加官方randaug方法M10 + official_init: 75.21/92.26
614: 官方efficientnet + swish + cv2的transform加官方randaug方法m9nocutout + official_init: 75.02/92.2
614: 官方efficientnet + swish + cv2的transform加官方randaug方法m9nocutout-mstd0.5 + official_init: 75.17

614: 再用自己的模型: 
使用efficientnet-pytorch: 73.45/91.16

看是否能复现resnet的结果: 
    70.61/89.42
    再来: 72.27/90.77
    去掉ema: 73.28/91.36
    使用自制resnet: 74.04/91.66
    使用初始化: 74.09/91.64
    bn加上eps: 74.17/91.74
    去掉warmup: 73.95/91.74
    加上sync-bn呢: 40个epoch才21，有问题
    换上apex的syncbn和DDP: 貌似可以了，是sync-bn使用位置的问题: 73.88/91.53
    使用pil的函数去load呢: 72.62/91.12
    使用inter_cubic的interpolation: 

    使用pycls里面的opt参数: 74.15/91.76
    去掉batch_sampler: 74.15/91.76

按官方的做: 
    bn加正常的wd, 不加warmup， 
        bs=1024+lr=0.1: 71.66/90.35
        bs=1024+随batchsize变化: 74.18/91.74
        使用example里面的dataloader: 76.82/93.44
opencv vs pil:
    pil: 76.75:
    opencv: 
        直接指定256: 76.13
    opencv的resize + pil的其他: 
        linear: 76.12
        cubic: 74.96

#####
1. 前面不行的原因是
    * pil reisize的interpolate的方法需要改成bilinear才行，如果怎么都不说只使用默认的话，怕是不太行，torchvision.transforms里面就给设成了bilinear
    * opencv的resize的h和w搞反了
####

resnet50: 
    没正则: 76.53/93.41
    加ra:
        prob=0.5: 77.72/93.85
        prob=1: 77.37/93.66
        prob=0.1:

    加上ema: 
        ema在model之后: 不稳定了
        ema在ddp之后:  不稳定了
        ema不带fp32的: 
        单独model的ema呢: 稳定了
        看样子ema实现是有点问题的

    加上lbsmooth:
    77.71/93.72


efficientnet: 
    baseline: 75.98/92.83/76.93/93.26
    去掉randaug: 75.70/92.70/76.89/93.29
    refactor之后的: 75.59/92.65/76.76/93.20
    refactor之后的, ra=2,10: 75.63/92.54/76.83/93.17
    refactor之后的, ra=2,9, p=0.1: 75.93/92.87/77.00/93.51
    refactor之后的, ra=2,9, p=0.5: 76.02/92.78/77.25/93.44 -- 不知道是啥
    refactor之后的, ra=2,9, p=0.5: 76.27/92.99/77.18/93.47
    refactor之后的, ra=2,9, p=1: 75.96/92.76/76.99/93.29
    结论，还是设成p=0.5吧, 
b1:
    ra=2,9, p=0.5: 77.26/93.42/78.1/93.79
    去掉ra: 76.79/92.93/77.53/93.45
    修正eval, ra=2,9, p=0.5: 78.21/94.09/79.27/94.53

b1使用cuda版的swish: 78.17/94.14/78.99/94.51 -- 差不太多，有点跳

看一下三个版本的label smooth的速度
effnet-b1: 
    v1: 142s
    v2: 153s
    v3: 136s

看修正pca_noise之后是否变快了: 
修改后: 131s
修改前: 139s 

effnet-b1: mixup+lbsmooth(0.1):
    alpha=0.2: 76.09/93.03/77.10/93.54
r50 + mixup:
    baseline, 无正则: 63/85.26/73.93/91.61 
    换成nn.cross_entropy，和1.6的原生tp16: 60.73/83.52/73.83/91.76
    换成nn.cross_entropy，和1.6的原生tp16，用回原来的scheduler: 76.57/93.32/76.85/93.46
    alpha=0.4, lbsmooth=0.0, no-ra: 77.49/93.86/77.55/93.93
    alpha=0.4, lbsmooth=0.1, no-ra: 77.64/93.91/77.72/93.96
    alpha=0.4, lbsmooth=0.1, ra2-9: 77.46/93.84/77.51/93.80

单上cutmix: 77.4/93.68/77.45/93.73
改变lam顺序，并且不要max: 77.67/93.88/77.77/93.93
r50改成cosine-lr + 300ep + cutmix 原版: 78.84/94.35/78.96/94.39
r50改成cosine-lr + 300ep + cutmix 任意大小位置的: 78.66/94.3/78.82/94.39
使用自己的cdataloader，另外弄一个train.py

refactor1: 77.73/93.87/77.78/93.94 
refactor2: 77.61/93.9/77.79/93.91 

r50, 使用自己的cdataloader:
baseline: 76.55/93.21/77/93.37
+ra-2-9: 77.45/93.66/77.44/93.67
+pca-noise: 77.44/93.62/77.49/93.67
所有的double都换成float看能加速不: 好像没有太大区别

r50
用原来的dataloder + cuda_one_hot: 76.64/93.27/76.97/93.47
用cpp的image reader + cuda one hot: 可以，但是太慢了

再来一遍r50, 300x, cosine, lbsmooth:
lbsmooth + mixup + cutmix + ra:  78.09/94.06/78.14/94.04
lbsmooth + cutmix + ra:  78.83/94.36/78.84/94.35
lbsmooth + ra: 78.14/93.96/78.27/94.01 
lbsmooth + cutmix:  78.58/94.30/78.77/94.37


500ep: lbsmooth + cutmix + ra: 79.11/94.47/79.09/94.47
500ep: lbsmooth + cutmix + mixup + ra: 78.48/94.21/78.53/94.25

200ep: lbsmooth:
200ep: lbsmooth + ra:
200ep: lbsmooth + mixup:
200ep: lbsmooth + cutmix:


r50，看先relu再bn, bottleneck不变: 77.05/93.40/77.06/93.43
r50，看先relu再bn, bottleneck改成两个分枝先相加再relu，再bn, shortcut去掉bn: 不收敛 -- 这个最省内存6529M
r50，看先relu再bn, bottleneck改成两个分枝先相加再relu，再bn, shortcut加上bn: 不收敛 -- 这个最省内存7121M
r50，看先relu再bn, bottleneck改成两先relu再相加再bn : 不收敛 -- 内存为8733M
r50，看先relu再bn, bottleneck改成两先relu-bn再相加 -- 没必要
r50，看先relu再bn，第一个conv-bn-relu-maxpool改成conv-relu-maxpool-bn: 77.06/93.48/77.08/93.50 --  内存是7013M
能不能像mbconv那样，去掉带conv的shortcut，然后去掉最后的act


===
关于frelu-r101: 
    eval时间比r101的4m43s多一些，大概是6m9s，内存也多,从4400M到6000M，但是参数量的话，从171M到174M

    关于训练时: 
        r101内存5949M-101s, 
        frelu内存11589M-174s,
        frelu+conv-max-pool-bn-relu内存11445M-171s:

关于dropblock: 看resnet是否应该像tpu里面那样，在shortcut的地方加上dropblock，是否应该加上drop-connect

spinenet-49s-100ep: 73.18/91.59/73.18/91.59
目标是49: 77/93.3

ushape_effnet-b0，像spinenet那样merge所有feature: 
    直接effnet+fpn: 74.84/92.23/74.9/92.24
    effnet+4xconv+fpn: 74.88/92.03/74.93/92.12
    直接effnet+fpn, 后面加一个4xconv: 75.87/92.66/75.92/92.73 -- 这个有用

taylor-softmax: -- 都是没有lbsmooth的，所以跟model_zoo不太一样
    r50-baseline: 122s/7155m: 76.22/93.17/76.06/93.01
    r50-taylor-softmax:122s/7159m: 76.76/93.40/76.18/93.33
    effnet-b0-taylor_softmax: 75.10/92.61/75.03/92.61
    effnet-b0-baseline: 75.40/92.54/75.44/92.56
    effnet-b0-taylor_softmax-taylor_se: -- discard, se使用的是sigmoid
    effnet-b0-large_margin_v3: 76.10/92.45/75.90/92.38
    上面做完了，再试一下large-margin吧，看是在imagenet上也有用


repvgg-a2: 75.26/92.52/75.53/92.6
repvgg-b0: 74.26/91.96/74.39/91.88
去掉warmup, repvgg-b0:
    74.24/91.75/74.13/91.74
fix掉dataloader的random问题之后再来:

b0: 71.7/90.07/71.82/90.12
官方脚本训练: 72.40 -- 目标值 
使用官方cosine: 71.75
官方脚本使用自己的transformers: 71.82
自己脚本使用官方的transformers: 使用opencv读图: 71.92, 使用pil读图: 72.05
optimizer使用官方的: 72.23/90.66/71.60/90.30
修正自己的hflip: 72.28/90.68/71.63/90.33
使用bs=32的:72.03

a0的fp32:
a1的: bs=256,fp16: 74.46/91.78/: bs=32,fp16: 74.03/91.73, bs=32,fp32: 74.09
a2的: 76.31/93.05/76.4/93.05 -- 再来bs=32,fp16: 76.04, 调整get_params再来76.38, 
    fp16+bs256: 76.07 -- 76.08
    fp16+bs32: 76.17
    fp32+bs32: 76.44
结论: 
    * 一定是fp32+bs32才能复现，其他配置不能复现
    * 要把nesterov换成false才行
a1, fp32+bs32: 74.21
b0, --目标75.14
    fp16+bs256: 74.9, 
    bs=32+fp32: 74.96, 
    nesterov=False: 75.22 
    恢复bs=256,fp16加上nesterov=False: 74.87
    fp16+bs32, nesterov=False: 75
    fp32+bs32, nesterov=False: 75.22/75.03
    
b1, fp16+bs128:  -- 目标78.37
    fp32+bs32, nesterov=False: 
    
b2, fp16+bs128: -- 目标78.78


eval-epoch改成1，看最大的那个epoch是多少:
    改成inter-linear: 72.06
修正hflip使用自己的optimizer: 
    dim==1的不带wd: 72.15/72.32
    都带wd: 69.97/69.99
    修正flip, 使用官方optimizer的逻辑，fp16: 72.18/72.22
    上面的使用fp32: 72.33/

修正resnet-d:
    r50_slim:
    r50_slim+ge:
    r50_d:
    r50_d_se:
    r50_d_slim:
    r101_d:
    r101_d_se:
    r34:
resnet-rs的trick都加上:
看lookahead + adamw是否能加快收敛?

densecl, resnet50, 100ep, 看sgdm_adamw+lookahead是否能加快收敛:
    sgd的: 82.7/58.5/65.6
    adamw+lookahead: nan
    sgdm_adamw+lookahead: nan
        lr=0.03:
    


rednet的involution:
    rednet38_ibn_b: 75.65/92.63/75.77/92.67
    resnet38_ibn_b: 74.58/92.16/74.67/92.19
        74.8/92.15/74.9/92.23
    resnet38_ibn_b_sepconv: 75.06/92.25/75.13/92.27

    rednet38: 75.68/92.45/75.73/92.53
    resnet38: 75.95/92.81/76.02/92.89
        75.64/92.82/75.80/92.83
    resnet38_sepconv: 75.28/92.38/75.29/92.50
    结论: 
        involution在分类上有的时候有用，有的时候没用，加上ibn之后有用，普通的cnn没用
        更新: 基本上没用，分类和分割上都没用

把lbsmoot/ra都加到config里面去


======
model_zoo:
effnet-b0: 76.03/92.84/75.96/92.78
effnet-b0-bn0: 76.02/92.95/75.75/92.83
effnet-b0+ra+200ep: 76.84/93.34/76.91/93.34
effnet-b0-conv: 76.19/92.29/75.87/92.31
effnet-b0-lite-conv:  75.90/92.36/75.88/92.34 
effnet-b2: 78.66/94.41/78.69/94.38 -- 目标是79.8
effnet-b2-lite: 77.69/93.86/77.65/93.85 
effnet-b2-conv: 79.95/94.80/80.00/94.82
effnet-b2-lite-conv: 78.90/94.33/78.93/94.34 
再来effnet-b2: 78.79/94.26/78.77/94.28  -- 差不多，这个不保存
effnet-b0-lite: 74.64/92.08/74.62/92.07
effnet-b4: 81.04/95.57/81.08/95.59 -- pycls(78.4), official(82.5)
effnet-b6: 82.07/95.92/82.06/95.94 -- pycls(没有), official(84.00-带aa的)
r50: 77.19/93.66/76.72/93.49
r50不带bn0的: 77.3/93.77/76.56/93.43
r50_lsr_ra_200ep: 78.34/94.11/78.3/94.2
r101:78.50/94.33/78.43/94.42
r101+frelu: 78.88/94.35/79.06/94.54
r101, frelu, conv-maxpool-bn-relu: 78.71/94.26/79.16/94.4 -- 上次跑到这了
pa-r50: 76.93/93.47/76.69/93.43
pa-r101: 78.37/94.11/78.04/94.11
se-r50: 78.74/94.31/78.81/94.40
se-r101: 79.78/94.89/80.02/95.01
ibn_b_r50_ra_lsr_200ep: 78.4/94.18/78.46/94.21
pa-se-r50:
pa-se-r101:
dynamic-conv-r50:
xception-41(deeplab): 80.50/95.20/80.54/95.23 -- 目标是79.55/94.33
xception-65(deeplab): 81.14/95.49/81.23/95.53 -- 目标是80.32/94.49
xception-71(deeplab): 81.28/95.66/81.29/95.70
spinenet-49s-200ep: 74.48/92.10/74.50/92.15
spinenet-49-200ep: 77.38/93.72/77.51/93.74
ibn_b_resnet101:
ibn_resnet101d-a:

带cutmix的:
    ibn_resnet50-a: 77.27/93.78/75.52/93.56
    ibn_resnet50-b: 75.45/92.83/74.77/92.44
不带cutmix的:
    ibn_resnet50-a: 77.42/93.63/77.49/93.74
    ibn_resnet50-b: 73.56/91.49/72.87/91.04
    ibn_resnet50-b, fix_maxpool: 74.08/92.04/73.79/91.75
    ibn_resnet101-a: 78.61/94.36/78.84/94.51

ibn_a_resnet101-blur: 78.58/94.34/79.06/94.64
ibn_b_resnet101-blur: 78.65/94.39/79.11/94.57
resnet101-blur: 77.79/93.83/78.28/94.1 

上面有可能要重来: -- 修正fhlip之后重来
resnet50, on_bn0: 6128: 76.73/93.29/76.46/93.28
resnet50_blur:77.11/93.56/77.57/93.69
resnet101, no_bn0: 619: 77.80/93.89/77.92/93.96
resnet101_blur: 619: 77.92/94.06/78.51/94.25
ibn_b_resnet101-blur: 6128: 78.22/94.01/78.61/94.36 -- 再eval一下
ibn_b_resnet50-blur: 77.18/93.60/77.61/93.80
ibn_b_resnet50: 73.28/91.40/72.51/90.84 -- 再来
ibn_a_resnet50: 77.40/93.82/77.52/93.76
ibn_a_resnet50-blur: 6128: 77.62/94.01/78.20/94.19
ibn_a_resnet101: 78.51/94.23/78.93/94.44
ibn_a_resnet101-blur: 78.45/94.36/79.21/94.62

ibn_b_resnet50: 
    正常: 73.28/91.40/72.51/90.84 -- 再来
    去掉last_bn:
    1024-s32: 72.66/90.98/71.65/90.39
    1024-s16: 75.71/92.82/75.50/92.72

不用重来的:
ibn_b_resnet38: 75.65/92.63/75.77/92.67





把effnet的fc改名成classifier，统一一下: 

改成自动根据gpu数来调学习率的

试一下batch augmentation，同一张图做M次aug，组成一个batch这种。

weight_align_r50: discard
weight_align_r101: 80ep后top1到70，然后loss变成nan
r50使用hs-r50的300ep训练方法:
hs-r50:


改成data和datasets分开的

结论: 
    1. cutmix跟ra作用重合
    2. cutmix和mixup一起用效果变坏
    3. effnet-b6训练了44天，xception65只要3天，所以大模型的话，effnet不是很好，不发xception
    4. xception41只比xception65低0.6个点，xception65只比xception71低0.1个点，所以最好的是41
    5. taylor-softmax的loss比exp-softmax的大一点，所以taylor引入的正则更强一些
    6. ema和random-depth不适合用来pretrain，加这两个之后在下游任务上会掉点
    7. 其他正则加强了之后要适当减小wd，比如同时有ra, rd, dropout的时候要减小wd

effnet改成把conv_out从backbone里面拿出去:  
effnet把conv和conv_ws版本分开: 


改变mixup实现，变成model, crit，都放进去，只mix loss的: 


done 尝试pytorch1.5的fp16，看能不能去掉apex

加上梯度裁剪: 

eff加上last-bn: 
eff换成relu试试: 

    dropconnect长到0.3: 
    固定dropconnect=0.3:

然后试一下logit regularization: 

再试一下resnet50的大batch

试一下loss的boostrap:

使用effnet里面的参数和optimizer看能不能到差不多的: 


effnet换成sgd能行吗

看是否把se加上temperature，并且前10个epoch从T=30一直anneal到1这样会加快收敛。  


mixup: 同batch mix和异batch mix: 直接上lambda还是1-lambda, 使用自定义的ce还是nn.ce 还是sigmoid
cutmix:
mixup + cutmix: 

看能不能再refactor一下，能不能让不同模型都兼容，而不是要到代码里面去注释一部分这样: 

检查eval的center-crop是否正确: 

官方好像是warmup_epoch=3, 我这里一直是5: 

都可以了的话，再试一下先crop再resize到224:


discard 不要改。 eff的还是把drop path rate改一下?

611: 
如果rmsprop失败了的话, 改回sgd-momentum: 
    + lr-scheduler不考虑warmup:
    + init:
    + wd位置: 
    用回cosine:
    用回ce: 

看lr是否是sqrt的关系, 而不是linear的关系?

看一下要不要在保存模型的时候同时也保存一个backbone的，保存一下专门用来抽feature用的模型，并且实现加载功能


discard 使用lmdb看能加速不

原版里面是使用的l2loss在每一个param上，并且对bn的所有参数都不加l2

wd=0, 然后使用手动weight decay的方式加wd, 像adam一样: 
直接计算L2-loss(按tf里面的方法来): 
直接用wd作为参数weight-decay: 
用wdxlr作为参数做weight decay: 

初始化方法: 

加wd的位置: 都加: conv-bias不加, conv和bn的bias不加(相当于只给weight加, bn也有weight), bn都不加



mixup: 这这里的mixup是每个不同的样本有各自的lam， 另外， lam是max(lam, 1-lam)这样确定的。 再试试shuffle单个batch的版本和两个batch合并的版本。  
而且好像不是shuffle， 而是imgs 和imgs[::-1]做的mixup. 

====
先试pre-act-r50:
baseline: 使用r50训练参数: 
    76.76/93.32/77.07/93.57
    mem=6600m, time=120s
    感觉有点跳，95ep的时候已经上77了，然后最后5个ep之后又掉下来了

前面加一个BN:
    76.75/93.31/77.15/93.61
    mem=6600m, time=125s

修正cosine-lr重新来:  
一开始改成conv-bn-relu-maxpool，再加后面这些
    mem=6621m, time=122s:
    76.92/93.50/76.70/93.39

baseline:
    mem=6645m, time=120s: 76.97/93.56/76.79/93.49
baseline+bn: 
    77.16/93.64/76.69/93.46

resnetv1，并且修正最后的stride=1的问题，改成dilation那种:
    mem=7917m, time=119s: 76.99/93.53/76.80/93.43
resnetv1前面加bn:
    mem=7917m, time=119s: 77.2/93.55/76.76/93.51

加上se之后看是否需要加上last-bn:
不带last_bn: 78.11/94.05/78.46/94.24
带last_bn: 

pa-resnet50-se:78.27/94.16/78.11/94.17

pa-resnet101: 78.05/94.07/77.91/94.13

r101: 78.51/94.29/78.21/94.14

pa-resnet101-se:

618把se-block也加进去: 77.15/93.72/77.13/93.67


看是否需要实现一个aff-atten的: 78.75/94.37/78.62/94.46


都试完，再看是否需要把shortcut加上avg-pool这种:

别忘了再来一个r50的，看r50的内存啥的，还有效果啥的都咋样

refactor:
resnet和 resnet-slim分开的，不一定分文件，但是使用两个类
done 单弄一个resnet-base文件吧，里面是bottleneckv1/v2, create_layer啥的，然后把其他的单拿出来
都完事了之后，弄一个hub,把自己的权重都推上去
然后都改成保存的时候保存一个backbone的state_dict，再保存一个model的:

====
617:
先mean再加起来:
1xlr: 不行了5个epoch之后nan爆掉
0.5xlr: 72.66/90.7/73.25/91.01

全nearest到最大的logits再加起来
加mean: nan
加3x3conv到256最后再mean: 72.53/90.47/0.1/0.4
加3x3conv到1024最后再mean: 73.51/91.32/0.1/0.6
加1x1conv到1024最后再mean: 73.66/91.28/0.1/0.6

全nearest到最大的logits再concat起来: 
再3x3conv到256最后再mean: 72.66/90.65/0.1/0.5
再3x3conv到1024最后再mean: 73.89/91.26/0.1/0.5
再1x1conv到1024最后再mean  discard

给前面的stage加一些conv，前几个stage也像最后一个一样，加一个conv_head, 都是x4倍的:
再3x3conv到1024最后再mean: 还是会变0.1
只使用最后的8x的feature，并且修正fpn的不对的地方: 还是会变0.1
三个feature先mean，再加起来，再加nn.linear: 还是变0.1  

改成sgd+cosine: 
三个feature先mean再加起来，然后nn.linear到n_classes:
    70.46/89.28/70.54/89.35
三个feature分别加几个conv到32x再相加，再加conv，再mean啥的: 
    71.46/89.75/71.48/89.77
把学习率调大(上面这个是1.6e-2):
    lr=1.6e-1: 74.5/91.88/0.0/0.4

上618的参数0.2(128)去掉ws, 使用swish, ep=100, 去掉ra/lbsmooth/mix:
    8x, 16x分别下采样，再相加，再conv，再mean/linear
        73.97/91.7/74.48/91.92
    去掉pyramid里面额外的conv:
        73.92/91.66/74.24/91.89

8x,16x分别downsample 2/1次，再相另: 73.92/91.66/74.24/91.89
直接都nearest到最大，然后相加再conv_out: 74.23/91.87/74.52/92.08
直接都nearest到最大，然后concat再conv_out: 74.12/91.77/74.43/91.95

直接都nearest到最大，3x3conv然后相加再bn+act:



因为effnet最后是使用1x1conv放大到x4的channel数，然后再分类的，所以这里也可以放大最后的channel试试

====
618:
pretrain-backbone, cosine-lr, rms-prop, ep=150, ra, lbsmooth=0.1:
effnet-swish: 75.72/92.62/75.64/92.64
effnet-hswish: 75.88/92.66/75.97/92.60
effnet-relu: 75.88/92.68/75.87/92.62
调整之后的effnet-relu: 
    nn.conv: 75.86/92.71/75.81/92.71
    wsconv: model的ema: ema变成0了
            state_dict的ema: naive变成0了。
            如果不行，去掉convws的state_dict，或者在里面加一个参数啥的
    改用sgd + cosine lr + ws:  65.79/86.49/65.82/86.51 --> n1
    看样子, rmsprob和wsconv是冲突的.  
    sgd + cosine lr + nn.conv: 65.72/86.57/65.75/86.58

    像regnet里面那样的参数:  
    sgd + cosine lr + nn.conv, lr=0.1/(im_per_gpu=128，wd=5e-5): 
        72.18/90.62/1/5
    看一下变成0的ema是否是因为出现了nan -- 不是，没有，别瞎说
    再来上面的，然后改成100ep，像pycls那样:  70.23/89.47/70.23/89.37
    去掉ws看是否有用: 73.38/91.37/73.41/91.37

nn.conv2d+swish+pycls超参, +ra, +lbsmooth: 74.96/92.07/74.86/92
nn.conv2d+swish+pycls超参, -ra, -lbsmooth: 75.46/92.46/75.50/92.53 -- 达到pycls的75.1了
nn.conv2d+hswish+pycls超参: 75.28/92.46/75.45/92.52

nn.conv2d+hswish+pycls超参, +ra, +lbsmooth, ep=200: 76.29/92.87/76.27/92.88
nn.conv2d+hswish+pycls超参, +ra, +lbsmooth + cutmix, ep=400: 73.71/91.81/73.64/91.8
nn.conv2d+hswish+pycls超参, +ra, +lbsmooth + cutmix, ep=200: 
