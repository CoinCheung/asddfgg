
baseline: 
69.02/89.11

加上scheduler
68.89/88.96

label_smooth: 
68.54/88.79

拿到611上，并且修正crop_eval:
68.14/88.42

使用apex的fp16: 

apex的fp16并且bs为原来2倍: 

wd改成1e-5,
wd改成2e-5,

加wd的位置: 都加: conv-bias不加, conv和bn的bias不加(相当于只给weight加, bn也有weight), bn都不加

初始化: 

改用rms-prob.

再来scheduler， 
warmup: 

eval的时候是center-crop吗: 

其他scheduler: 
wd改用l2，不用wd: 

先read， 再randomresizecrop， 再flip， 最后再autoaug

mixup: 这这里的mixup是每个不同的样本有各自的lam， 另外， lam是max(lam, 1-lam)这样确定的。 
而且好像不是shuffle， 而是imgs 和imgs[::-1]做的mixup. 

aug的时候，先auto-aug再random-crop: 

有空试一下把acc和ema-acc交换位置， 看结果还一样不。

