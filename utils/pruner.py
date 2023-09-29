
import torch
import math


from .arg import is_prunable_module



def applyMask(model, masks):
    model_remained = 0
    model_total = 0

    for name, m in model.named_modules():
        if is_prunable_module(m):
            mask = masks[name]

            total = mask.numel()
            model_total += total

            remained = int(torch.sum(mask))
            model_remained += remained

            m.weight.data.mul_(mask)
            if m.weight.grad is not None:
                m.weight.grad.data.mul_(mask)

    keep_ratio = model_remained / model_total

    return model, keep_ratio

def _get_w(model):
    w = []
    for k, m in enumerate(model.modules()):
        # print(k, m)
        if is_prunable_module(m):
            w.append(torch.reshape(m.weight.data.abs(), [-1, 1]))
    return torch.cat(w, dim=0).squeeze()

def get_prune_mask(model, prune_rate):
    w = _get_w(model)

    w_sort, _ = torch.sort(w)
    prune_idx = math.floor(w.shape[0] * prune_rate)
    prune_thresh = w_sort[prune_idx]

    mask = {}
    for name, m in model.named_modules():
        if is_prunable_module(m):
            mask_keep = (m.weight.data.abs() > prune_thresh)
            mask[name] = mask_keep

    return mask


def mag_prune(model, prune_rate):
    w = _get_w(model)

    w_sort, _ = torch.sort(w)
    prune_idx = math.floor(w.shape[0] * prune_rate)
    prune_thresh = w_sort[prune_idx]

    mask = {}
    for name, m in model.named_modules():
        if is_prunable_module(m):
            mask_keep = (m.weight.data.abs() > prune_thresh)
            mask[name] = mask_keep
            m.weight.data = m.weight.data.mul_(mask_keep.type(torch.float))

    return model, mask


def print_mask_statistics(mask, logger):
    pass
    CNT = 1

    keep_param_model = 0
    total_param_model = 0

    for name, m in mask.items():

        keep_param_mask = m.type(torch.int).sum().item()
        total_param_mask = m.numel()
        keep_ratio_mask = keep_param_mask / total_param_mask

        keep_param_model += keep_param_mask
        total_param_model += total_param_mask

        logger.info("LAYER %d(%s) : KEEP_RATIO = %.6f    NUM_PARA = %d    REMAINED_PARA = %d" %
                    (CNT, name, keep_ratio_mask * 100, total_param_mask, keep_param_mask))

        CNT += 1

    keep_ratio_model = keep_param_model / total_param_model
    logger.info("TOTAL MODEL : KEEP_RATIO = %.6f    NUM_PARA = %d    REMAINED_PARA = %d" % (
        keep_ratio_model * 100, total_param_model, keep_param_model))