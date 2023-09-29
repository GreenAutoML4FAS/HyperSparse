import torch

from utils import is_prunable_module


def lx_loss(model, p=1):

    w = []
    for m in model.modules():
        if is_prunable_module(m):
            w.append(torch.reshape(m.weight, [-1, 1]))
    w = torch.cat(w, dim=0).squeeze()

    p_norm = w.abs().pow(p).sum()

    return p_norm