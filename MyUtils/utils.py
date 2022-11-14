import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_randint(end_right, discard, nums):
    result_list = list(range(0, end_right))
    result_list.remove(discard)
    np.random.shuffle(result_list)
    return result_list[0:nums]


def normalize(a, axis=-1):
    x = a.clone()
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def featureprocess(f_cur, f_ref):
    weight = 0.8
    f_nums = f_ref.shape[0]
    object_nums = f_ref.shape[1]
    dim_f = f_cur.shape[1]
    f1 = f_cur.unsqueeze(0)
    f2 = f_ref

    cossim = torch.cosine_similarity(f1.unsqueeze(2), f2.unsqueeze(1), dim=3)

    maxidx = torch.argmax(cossim, dim=2)

    cos = torch.gather(cossim, dim=2, index=maxidx.unsqueeze(2))
    cos_soft = torch.softmax(cos, dim=0).squeeze(2)
    maxidx = maxidx.reshape(f_nums * object_nums)
    for i in range(maxidx.shape[0]):
        increment = int(i / object_nums) * object_nums
        maxidx[i] = maxidx[i] + increment
    f_temp = f2.reshape(f_nums * object_nums, dim_f)
    f_cos = torch.index_select(f_temp, 0, maxidx)
    f_cos = f_cos.reshape(f_nums * object_nums, dim_f)
    cos_soft = cos_soft.reshape(-1, 1)
    f_aug = (f_cos * cos_soft.reshape(-1, 1)).reshape(f_nums, object_nums, dim_f)
    f_aug = f_aug.sum(dim=0, keepdim=False)

    f1 = weight * f1 + (1 - weight) * f_aug

    return f1
