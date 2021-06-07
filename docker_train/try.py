
import torch
import torch.nn as nn
import torch.nn.functional as F
from cbl_models.hrnetv2 import HRNetBackbone


#  if __name__ == "__main__":
#      net = HRNetBackbone()
#      inten = torch.randn(1, 3, 224, 224)
#      outs = net(inten)
#      for o in outs:
#          print(o.size())


import cv2
import numpy as np


arr = np.random.randn(4, 4)
print(arr.shape)
re = cv2.resize(arr, (8, 8))
print(re.shape)
