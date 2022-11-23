import torch
import torch.nn.functional as F
import numpy
import math
from MyUtils.utils import featureprocess

a = torch.arange(0, 24).view(2, 3, 4).type(torch.float32)

index = torch.arange(1)
feat_topk = torch.index_select(a, 0, index)

print(feat_topk)
feat_topk[0,0,0] =100
print(a)