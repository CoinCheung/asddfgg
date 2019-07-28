
lr=1e-2/1e-5, wd=1e-4, n_epoch=20/120, bs=64
model_final_1.pth: 82.82
未完全收敛，loss还在下降

lr=1e-2/1e-5, wd=1e-4, n_epoch=20/200, bs=64
model_final_2.pth: 84.95

dynamic-refine:
使用model_final_2.pth去refine


======
naive: 
bs=256, lr=2e-1, wd=1e-4, n_epoch=20/200, resize+crop: 84.06/85.18/83.32

dynamic-refine: 83.06

online-refine:

=====
再调naive: 
bs=256, lr=2e-1, wd=5e-4, n_epoch=20/200, resize+crop: 

from scratch: 83.54
第一个conv换成3x3 stride=2: 84.23
第一个conv换成3x3 stride=2，去掉maxpool: 88.73(dyna:89.06)
第一个conv换成3x3 stride=1: 90.52
label-smooth: 90.77
wd=5e-4: 91.86(dyna: 92.22)

====
以上，外加refactoring: 92.06

使用last-residual-bn-gamma=0: 92.17
warmup改成10epoch: 92.10
warmup改成5epoch: 91.87
shortcut换成3x3 conv: 91.54
cos到0: 91.80

====
warmup改用10，shortcut还是1x1conv, cosine到1e-5了

使用pre-act结构:
    带last-gamma: 54.29
    不带last_gamma: 91.58

======
以上，但是换成cifar专用模型resnet20-v1: 
pre/non-pre:
    pre: 87.79
    non-pre: 87.79
    都差不多，所以先使用non-pre的了。 
wd=1e-4/5e-4:
    1e-4: 85.61
    5e-4: 87.79
    还是大一点好，看样子还是有过拟合的。
cos=0/1e-6/1e-5
    0: 87.52
    1e-6: 87.32
    1e-5: 87.79
    1e-4: 87.70
    看样子都差不多的，只要别太小了就行。
试一次label-refinery: 
    87.70/88.70/88.17/88.98/89.47 
    确实有帮助，能提一个点的样子
试一次换成56层的: 
    89.84
去掉label-smooth:
    87.10
    看样子还是有用的
warmup-epochs: 
    1: 87.79
    2: 87.92
    5: 87.75
    10: 87.96
    看样子对这个模型来说，多少个warmup epoch没啥区别 
bn的momentum设成
    不指定: 88.44
    默认0.1: 87.28
    0.01: 87.34
    1/bs^2: 87.64
在residual里面加上dropout: 
    non-pre-act: 88.06
    pre_act: 87.88/88.24
    看起来都差不多，加上也无所谓
channel-drop: 
    随机drop最多两个channel: 
        200epoch: 84.65 
        400epoch: 84.66
        lb-refine: naive=50，其他200: 82.65/81.36/80.32/79.43/77.79/77.46/76.17/76.44 
        lb-refine: naive=50，其他50: 82.65/80.54/78.86/77.72/75.96/75.51/74.93/94.15 
        说明: 
            1. 200 epoch足够了，多了也这样
            2. lb-ref这个需要初始的模型是一个收敛得比较好并且又没有过拟合严重的模型才有效，不然就会越来越差
    随机drop最多一个channel: 
        200epoch: 86.60
exp-warmup/linear-warmup:
    linear: 87.50
    exp: 87.25
    感觉都差不多，还是使用linear吧
使用opencv的pre-process: 算了，默认的cifar只支持PIL
pad=4再crop:
    92.31
    说明之前是预处理做得不对
使用cifar的norm做归一化:
    92.17
    说明都差不多其实

再加channel-drop: 
    90.16
    加上就不行，还是别加了

再来baseline: 92.31

====
以上，换成pre-act, warmup-start-lr=1e-5, no lb-smooth:
baseline: 
    mul: 91.31
    cos: 92.28
    结论: 确实cos更好一些
加上lb-smooth: 
    0.95/0.00005: 92.06
    0.9/0.0001: 92.18
    反而更差了一些
后面先不加lb-smooth了:

dropout: 
    都不加: 92.46
    最后fc加上dropout: 
        前面加: 91.86/92.35
        后面加: 91.95/92.38
        加前面还是后面都差不多，但是都比不加差一些。
        所以后面先都不加了
    residual dropout的位置: 
        pre-act加dropout在最后一个conv前: 
        0.3: 92.28
        0.1: 92.32
        non-pre-act加dropout在最后一个conv前:
        看样子加dropout就是不太行，所以先不加了

sgd-restart:
    len=10, mult=2, max=200: 81.63
    len=10, mult=2, max=160: 92.19
    len=150, mult=1, max=160: 91.90
    len=38, mult=1, max=200: 91.49
    len=10iter, mult=2, max=200，按iter调:90.89
    len=10epoch, mult=2, max=200，按iter调: 76.35
    len=10epoch, mult=2, max=160，按iter调: 91.94
    len=10, mult=2, max=320, 按epoch调: 91.95
    len=310, mult=1, max=320, 按epoch调: 92.50
    len=40, mult=1.5, max=200, 按epoch调: 92.05

    按epoch调比按iter调的效果好，一定要完成整个cycle再结束的效果更好。 

    160个epoch的时候，使用restart更好，320个epoch的时候就不好了。可能是比较难调吧，大部分的时候都是差一些，只有小部分时候能比普通sgd强。 


sgd-restart加上lr-decay: 

eta:
    0: 92.46
    1e-5: 92.28
    1e-4: 92.84
    5e-4: 92.30
    看样子1e-4最好 

eta=1e-4, lb-refine: 92.76/92.76/93.19/93.22

mixup: 
    正常mixup 无lb-smooth:
        pre_act, alpha=1, eta=1e-6: 91.92
        pre_act, alpha=1, eta=1e-5: 92.24
        pre_act, alpha=1, eta=1e-4: 91.63

        pre_act, alpha=0.9, eta=1e-5: 92.20  
        pre_act, alpha=0.8, eta=1e-5: 92.47 
        pre_act, alpha=0.5, eta=1e-5: 92.47
        pre_act, alpha=0.2, eta=1e-5: 92.30

        * 看样子，使用不同的方法的eta应该是不同的，正常是1e-4，有mixup是1e-5

    mixup + lb-smooth:
        pre_act:
    mixup使用KL散度:
        eta=1e-5, alpha=1: 
    延长到320epoch:

使用自己实现的cross-entropy，都改成基于one-hot的形式的: 
baseline:  92.84

正常mixup:
    eta=1e-5, alpha=0.5: 92.79
    eta=1e-4, alpha=1: 91.98
    eta=1e-4, alpha=0.5: 92.83
    eta=1e-4, alpha=0.25: 92.62
    eta=1e-4, alpha=0.2: 92.75
    eta=1e-4, alpha=0.1: 92.88
    lb-refine, eta=1e-4, alpha=0.25: 92.88/92.82/92.70/92.68/92.22

    从过程看，还是1e-4+0.2效果更好些，虽然结果是0.5更好，看样子还是有过拟合，并且mixup有用，只不过模型太简单，没有那么严重过拟合，所以alpha也不用设那么大。

    如果有label refine的话，在refine的过程加mixup的提升效果不算明显，没有mixup能长0.5个点，加上mixup就只能长0.1个点的样子了。而且会越refine效果越坏 

kl-mixup:
    eta=1e-4, alpha=0.2: 92.75
    eta=1e-4, alpha=0.5: 92.83

    mixup的loss使用cross entropy和kl div是一样的，没区别

单lb-smooth，不加mixup:
    0.9/0.00005:
        lr=2e-1: 92.51
        lr=2.5e-1: 92.42 
        lr=1.5e-1: 92.24
    看样子lb smooth在这个模型里面是没什么用的，调学习率也没用。 

sgdr:
    len=10, mult=2, decay=1, epoch=320: 91.57 
    len=40, mult=1.5, decay=1, epoch=200: 92.05
    len=40, mult=1.5, decay=0.7, epoch=200: 92.39 
    len=40, mult=1.5, decay=0.5, epoch=200: 92.23 

    看样子sgdr就是不行,也不知道论文里面是怎么得到的结论,根本就不行.  


加dropout+大lr: 

mixup+label-refine:
    两个分别计算loss再加到一起:
    先把label加一起，再计算loss:

    实验发现，把label加一起，再计算loss的效果要好于把loss加一起计算的方法.

用回non-pre-act: 
baseline: 92.39
mixup: 
    alpha=0.1: 92.94
    alpha=0.3: 93.06
    alpha=0.5: 92.80


关于mixup: 
    1. 混合label比混合loss效果好
    2. 要根据模型复杂度以及是否过拟合确定alpha,大alpha会导致欠拟合
    3. 使用kl散度和混合label的ce的效果是一样的.
    4. 不同alpha对应的最优cos_eta不太一样,但是也不会差特别多.
    5. mixup对于non-pre-act的作用要比pre-act的作用更好.

sgdr: 不好使
lb smooth: 不好使


baseline: 92.39 
mixup, alpha=0.3: 93.06

label-refine: 
    全部mixup-alpha=0.3: 
        93.06/93.07/93.05/93.07/92.85
        93.33/93.18/93.00/92.81
    第一个是mixup-alpha-0.3, refine不加mixup: 
        93.06/93.24/93.20/93.45/93.37

    可以看出来,refine的时候不加mixup效果会更好


加dropout: 
    resnetv1: residual上加dropout
        p = 0.05: 92.59
        p = 0.1: 93.03
        p = 0.2: 92.29
    residual不加，最后的fc后面加: 
        p=0.1: 92.60
        p=0.01: 92.85 
    residual不加，最后的fc前面加: 
        p=0.2:  92.75
        p=0.1: 93.12
        p=0.01: 92.99

    resnetv2: residual中间加:
        baseline: 不加dropout，加bn: 92.87
        不加最后bn, p=0.1: 92.75
        加bn, p=0.1:  92.93
        加bn, p=0.2:  92.36


再试last_gamma，如果去掉这个会不会增加: 
    resnetv1: 
        不加last_gamma: 92.98
        加last_gamma: 93.23
        说明last gamma有用


试试leaky-relu: 
    resnetv1:
        slope=0.01, msra=0: 92.92
        slope=0.01, msra=0.01: 92.98 
    resnetv2: 
        slope=0.01, msra=0.00: 92.95 
        slope=0.01, msra=0.01: 92.65
        slope=0.00, msra=0.00:  92.93
        slope=0.00, msra=0.01:  92.58
    结论: leaky-relu是不行的，而且是怎么样都不行


最后ce之前加上BN:
    93.23
    确实高了不少，看样子最后加一个bn是有用的

bn+fc前的dropout: 
    93.09
    总体上比不加droput低了一点点，说明dropout确实跟bn有冲突


swish 激活函数: 
    92.82
改回relu吧，别的好像不好使: 
    93.23


使用ema更新模型参数: 
    alpha=0.9, sgd_wd=5e-4, lr=wd=0: 93.17
    alpha=0.9, sgd_wd=0, lr和wd加在ema上: 91.50 
    alpha=0.9, sgd_wd=5e-4, lr和wd加在ema上: 93.17
    不要cosine和warmup，单用ema: 88.24
    每次在ema的基础之上再学习模型参数: 
    ema反过来，让decay乘以新参数，

    ema会让模型更加稳定，更快收敛，但是不一定会更好，虽然也不会特别差
    ema不能代替lr scheduler，还是需要加上正常的lr scheduler才行


加上一个self-supervised的head，用来分辨旋转了多少度，看是否有帮助．
    连mixup一起加self-supervised: 89.06


label-refine: 
先训练一个early stop的，再一遍遍的label-refine:  -- 不好使，越来越差
给kl的label加上sharpen-temperature

使用cifar100: 


=====
使用label-refine，但是每次不要那么多epoch，每次50个epoch多几个cycle试试。
50/100/150/200: -- 不好使

弱监督: 每个类只取500张训练naive，然后后面不停的拿全部的label-refine

使用label-refinery 做reid的弱监督问题，只有market小train一下，然后直接拿到duke上不用label的refiner看是否有效。



=======
改成wide resnet, 使用使用resnet29-10试试: 

使用opencv的aug: 

使用lookahead: 
