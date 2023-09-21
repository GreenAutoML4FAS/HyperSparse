
import torch
import math
import torch.nn as nn


from utils import is_prunable_module




def hyperSparse(model, prune_rate):

    w = []
    for m in model.modules():
        if is_prunable_module(m):
            w.append(torch.reshape(m.weight, [-1, 1]))
    w = torch.cat(w, dim=0).squeeze()
    w_abs = w.abs()

    #calculate hs_loss
    w_sort, _ = torch.sort(w_abs.detach())
    prune_idx = math.floor(prune_rate * w_sort.shape[0])
    s = 0.6585 / w_sort[prune_idx]

    w_tanh = torch.tanh(s * w_abs)
    w_sum = torch.sum(w_tanh)

    hs_loss = w_abs * w_sum
    hs_loss = hs_loss.reshape(-1)
    hs_loss /= w_tanh.sum().detach()
    hs_loss -= w.abs()

    return hs_loss.sum()


def grad_HS_loss(model, prune_rate):
    pass
    #todo add derivative of HS-loss

