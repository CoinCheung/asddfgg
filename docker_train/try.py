
import torch
import torch.nn as nn
import torch.nn.functional as F
from cbl_models.hrnetv2 import HRNetBackbone


if __name__ == "__main__":
    net = HRNetBackbone()
    inten = torch.randn(1, 3, 224, 224)
    outs = net(inten)
    for o in outs:
        print(o.size())


