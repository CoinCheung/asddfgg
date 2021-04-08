
import torch
import torch.nn as nn

from cbl_models import build_model
from config.resnet101 import *


ckpt_path = './res/model_final_naive_r101.pth'

model = build_model(model_args)
sd = torch.load(ckpt_path, map_location='cpu')
model.load_states(sd)
model.eval()

bot = model.backbone.layer1[1]
bot.conv1 = torch.nn.utils.fuse_conv_bn_eval(
        bot.conv1, bot.bn1)
bot.bn1 = nn.Identity()
bot.conv2 = torch.nn.utils.fuse_conv_bn_eval(
        bot.conv2, bot.bn2)
bot.bn2 = nn.Identity()
bot.conv3 = torch.nn.utils.fuse_conv_bn_eval(
        bot.conv3, bot.bn3)
bot.bn3 = nn.Identity()
print(bot)


print('conv1')
w1 = bot.conv1.weight
b1 = bot.conv1.bias
wsum = w1.abs().sum(dim=(1, 2, 3))
bsum = b1.abs()
prw = (wsum < 1e-6).nonzero(as_tuple=True)
prb = (bsum < 1e-6).nonzero(as_tuple=True)
print(prw)
print(prb)
print(wsum.size())
print(bsum.size())

print('conv2')
w2 = bot.conv2.weight
b2 = bot.conv2.bias
wsum = w2.abs().sum(dim=(1, 2, 3))
bsum = b2.abs()
prw = (wsum < 1e-6).nonzero(as_tuple=True)
prb = (bsum < 1e-6).nonzero(as_tuple=True)
print(prw)
print(prb)

print('conv3')
w3 = bot.conv3.weight
b3 = bot.conv3.bias
wsum = w3.abs().sum(dim=(1, 2, 3))
bsum = b3.abs()
prw = (wsum < 1e-6).nonzero(as_tuple=True)
prb = (bsum < 1e-6).nonzero(as_tuple=True)
print(prw)
print(prb)
print(b3[prw])
