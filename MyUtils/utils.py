import numpy as np
import torch
import torch.nn as nn


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
    weight = 0.2
    f_nums = f_ref.shape[0]
    dim_f = f_cur.shape[1]
    f1 = f_cur.clone()
    f2 = f_ref.clone()
    a = normalize(f1)
    cossim = torch.Tensor(f_ref.shape[0], f1.shape[0], f2.shape[1])
    for i in range(cossim.shape[0]):
        b = normalize(f2[i])
        cossim[i] = 1 - torch.mm(a, b.permute(1, 0))
    maxidx = torch.argmax(cossim, dim=2)
    for k in range(f1.shape[0]):
        cossim_temp = cossim[:, k, :].squeeze()
        idx_temp = maxidx[:, k].reshape(f_nums, 1)
        sim = torch.gather(cossim_temp, 1, idx_temp)
        sim = torch.softmax(sim, dim=0)
        sum_f = torch.zeros(dim_f)
        for j in range(f_nums):
            sum_f = (1 / sim.sum()) * sim[j] * f2[j, maxidx[j, k]]
        f1[k] = weight * f1[k] + (1 - weight) * sum_f
    return f1
