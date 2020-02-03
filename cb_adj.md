
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
        改成450个epoch: 
        加上color-jitter(0.4)和random-erasing(0.2):
        warmup也算在real-iter里面: 
        warmup从1e-6开始: 
    proj_bn的gamma改成0: 

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

