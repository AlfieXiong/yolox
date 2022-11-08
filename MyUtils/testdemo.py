import torch
import torch.nn.functional as F
import numpy
import math
from MyUtils.utils import featureprocess

a = torch.arange(0, 32).view(2, 4, 4)

b = torch.arange(-26, -10).view(4, 4)

a = a.type(torch.float32)
b = b.type(torch.float32)
# b = b.type(torch.float32).unsqueeze(0)
# print(b.shape)
# output = F.cosine_similarity(a, b, dim=2)
# print(output.shape)

# cossim = torch.Tensor(a.shape[0], a.shape[1], b.shape[0])
# for i in range(cossim.shape[0]):
#     for j in range(cossim.shape[1]):
#         x1 = b[j]
#         x2 = a[i].squeeze()
#         c = F.cosine_similarity(a[i],b[j])
# print(cossim)

featureprocess(b,a)