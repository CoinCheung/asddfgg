

### Model structure
1. cifar or imagenet  
The resnet models for cifar and imagenet have different structures. Cifar models are like this: 

    **Downsample rate**: For imagenet, the downsample rate can be as large as 32, since the orginal images are large. In contrast, cifar images are too small, so it is better to not downsample too much. 

    **No. of Stages**: The cifar resnet generally have 3 residual stages with downsample rate of 4. It starts with a 3x3s1conv-bn-relu and then goes the residual stages without maxpool. The stride of the first residual stage is 1, while stride of the other two are all 2.

    **Width and Depth**: The channels of each residual stages are 16, 32, and 64, which makes the network narrower. As for the depth, it is controlled by the number of residual blocks in each stage. If the total layers of the model is `L`, then each stage should have `(L-2)/6` blocks. In the paper, `L` can be (20, 32, 44, 56, 110).

    **Only Basic Blocks**: Only basic residual block are used in the cifar models. No bottleneck blocks needed even when layers get 110.


2. pre-activation or post-activation  
pre-activation models start with a conv without bn and relu(if no maxpool followed), which are tucked into the residual block. 

    The pre-activation model should be end with a bn-relu module.


### Training Tricks
1. weight decay matters. A small weight decay may cause overfitting (the loss goes low but the test accuracy is not high). `5e-4` is better than `1e-4`.

2. label smooth helps a little. Maybe overfitting is always a problem with these small pictures. 

3. for the normal resnet, it is better initialized the value of bn gamma of each residual block to be 0. For the pre-activated resnet, it is harmful to initialize bn gamma to be 0.

4. as claimed in the paper of wide resnet, adding dropout to residual path can give a little help. In the implementation of the paper, dropout is added after the relu functions, and the first convolution should not have dropout.
