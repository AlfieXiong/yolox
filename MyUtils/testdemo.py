import torch

import numpy
import math

a = torch.arange(0, 24).view(2, 3, 4)
b = torch.Tensor(1, 3, 2)
b = a[1, :, 0:2]
print(b.shape)
b[0, 1] = 100
print(a)
print(b)
