
import torch
import math


from utils import is_prunable_module



def _get_w_tens(model):
    w = []
    for m in model.modules():
        if is_prunable_module(m):
            w.append(torch.reshape(m.weight, [-1, 1]))
    w = torch.cat(w, dim=0).squeeze()
    return w

def hyperSparse(model, prune_rate):
    w = _get_w_tens(model)
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
    w = _get_w_tens(model).detach()
    w_abs = w.abs()

    #calc s
    w_sort, _ = torch.sort(w_abs)
    idx = math.floor(prune_rate * w_sort.shape[0])
    s = (0.6585 / w_sort[idx]).item()

    #calculate HS-Gradients
    w_tanh_deriv = s / (torch.cosh(s * w_abs) ** 2)
    factor = w_abs.sum() / torch.tanh(s * w_abs).sum()
    grad = torch.sign(w) * w_tanh_deriv * factor

    #reshape to model_size
    CNT = 0
    w_grad = {}
    for name, m in model.named_modules():
        if is_prunable_module(m):
            loss_w = grad[CNT: CNT + m.weight.data.numel()]
            loss_w = loss_w.reshape(m.weight.data.shape)
            w_grad[name] = loss_w

            CNT += m.weight.data.numel()

    return w_grad