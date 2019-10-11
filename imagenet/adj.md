目标: 
    76.3/93.2

1. 第一次跑通
ce + relu + no ema + fp16
    65.53/86.42

2. 加上ema

关于前30个epoch没有效果: 1. 减小学习率为原来一半; 2. 使用sync-bn

3. 看使用h-swish能不能爆内存

看打印lr scheduler的问题是怎么来的 --> apex.  

看是否是imagenet的randomresizedcrop的问题, 原版用的是直接resize了

看一下warmup的start lr ratio

inplace-bn再加速

不行就上sgd: 

label smooth
1. 初始化方式 
1. last gamma = 0

2. 是不是跟bn有关, 换成sync-bn?

2. apex上fp16, 放大batchsize

2. 差不多的话, 把swish换成relu, 这样可以省内存, 还能快点
3. 如果还差不多的话，换成sgd+cosine试试

mixup:

1. label refinery

2. crop evel, flip eval

3. ohem 

4. lookahead


5. rmsprob 的alpha设成0


TODO: 
lr scheduler和fp16 的顺序
