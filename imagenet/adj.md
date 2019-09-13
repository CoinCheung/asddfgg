label smooth
1. 初始化方式 
1. last gamma = 0

2. apex上fp16, 放大batchsize

2. 差不多的话, 把swish换成relu, 这样可以省内存, 还能快点
3. 如果还差不多的话，换成sgd+cosine试试

mixup:

1. label refinery

2. crop evel, flip eval

3. ohem 

4. lookahead


5. rmsprob 的alpha设成0
