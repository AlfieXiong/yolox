import torch
import torch.nn.functional as F
import numpy
import math
from MyUtils.utils import featureprocess

a = torch.arange(0, 24).view(2, 3, 4)

b = torch.arange(0, 12).view(3, 4)

a = a.type(torch.float32)
b = b.type(torch.float32).unsqueeze(0)

c = F.cosine_similarity(b.unsqueeze(2), a.unsqueeze(1),dim=3)

print(a.shape)
print(b.shape)
print(c.shape)
