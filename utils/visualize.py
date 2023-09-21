import torch
import math

from matplotlib import pyplot as plt

from . import is_prunable_module

def plot_gradient(model, art_epoch, prune_rate):
    w = []
    g = []
    for m in model.modules():
        if is_prunable_module(m):
            w.append(torch.reshape(m.weight.data, [-1, 1]))
            g.append(torch.reshape(m.weight.grad.data, [-1, 1]))
    w = torch.cat(w, dim=0).squeeze()
    w_abs = w.abs()
    g = torch.cat(g, dim=0).squeeze()

    sort_w, sort_idx = torch.sort(w_abs)
    sort_g = g[sort_idx].abs()

    #plot gradient
    idx = torch.arange(0, w_abs.shape[0],1) / w_abs.shape[0]
    start_idx = math.floor(0.65 * w_abs.shape[0])

    x = idx[start_idx:].detach().cpu().numpy()
    y_w = sort_w[start_idx:].detach().cpu().numpy()
    y_g = sort_g[start_idx:].detach().cpu().numpy()

    fig, ax = plt.subplots(2)
    plt.grid()
    fig.suptitle(f"epoch: {art_epoch}")

    ax[0].plot(x, y_w)
    ax[0].set_title("sort magnitudes")
    ax[0].axvline(x=prune_rate, color='r', label='prune_thresh')
    ax[1].plot(x, y_g)
    ax[1].set_title("gradient")
    ax[1].axvline(x=prune_rate, color='r', label='prune_thresh')


    plt.show()
    plt.close()

    return w, g