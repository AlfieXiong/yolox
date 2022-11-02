import torch

import numpy
import math

a = torch.arange(0, 24).view(2, 3, 4)
print(a)
idx = torch.LongTensor([[[1], [1], [1]], [[1], [1], [1]]])
print(torch.gather(a, 2, idx))
