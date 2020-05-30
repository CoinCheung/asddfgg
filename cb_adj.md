
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
    * opencv的resize的h和w搞返了
####

resnet50: 
    没正则: 76.53/93.41
    加ra:
        prob=0.5: 77.72/93.85
        prob=1: 77.37/93.66
        prob=0.1:
merge之后的:

加上ema: 
    ema在model之后: 不稳定了
    ema在ddp之后:  不稳定了
    ema不带fp32的: 
    单独model的ema呢: 稳定了
    看样子ema实现是有点问题的

加上lbsmooth:
77.71/93.72

使用pycls里面的efficientnet和参数: 
换成自己的mbconv: 

再试试官方的ema
慢慢调自己的模型, 看能不能跟官方实现效果一样: 


mixup: 同batch mix和异batch mix: 直接上lambda还是1-lambda, 使用自定义的ce还是nn.ce 还是sigmoid
cutmix:
mixup + cutmix: 


检查eval的center-crop是否正确: 

官方好像是warmup_epoch=3, 我这里一直是5: 

都可以了的话，再试一下先crop再resize到224:


611: 
如果rmsprop失败了的话, 改回sgd-momentum: 
    + lr-scheduler不考虑warmup:
    + init:
    + wd位置: 
    用回cosine:
    用回ce: 

看lr是否是sqrt的关系, 而不是linear的关系?


使用lmdb看能加速不

原版里面是使用的l2loss在每一个param上，并且对bn的所有参数都不加l2

wd=0, 然后使用手动weight decay的方式加wd, 像adam一样: 
直接计算L2-loss(按tf里面的方法来): 
直接用wd作为参数weight-decay: 
用wdxlr作为参数做weight decay: 

初始化方法: 

加wd的位置: 都加: conv-bias不加, conv和bn的bias不加(相当于只给weight加, bn也有weight), bn都不加



再来scheduler， 
warmup, ratio啥的: 

eval的时候是center-crop吗: 

其他scheduler: cosine: anneal到0
wd改用l2，不用wd: 

先read， 再randomresizecrop， 再flip， 最后再autoaug

mixup: 这这里的mixup是每个不同的样本有各自的lam， 另外， lam是max(lam, 1-lam)这样确定的。 再试试shuffle单个batch的版本和两个batch合并的版本。  
而且好像不是shuffle， 而是imgs 和imgs[::-1]做的mixup. 

aug的时候，先auto-aug再random-crop: 

有空试一下把acc和ema-acc交换位置， 看结果还一样不。

