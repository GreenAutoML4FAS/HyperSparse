# ========== adopted from https://github.com/Eric-mingjie/rethinking-network-pruning ============
'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import torch
import math
import re

import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', "get_param_from_path", 'AverageMeter', 'ModelBuffer']


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

# ========== This function was modified to calculate overall zero params instead of conv zero params ============
def get_zero_param(model):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
            total += torch.sum(m.weight.data.eq(0))
    return total

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_param_from_path(relPathModel, name_load_model):
    namedParam = [x for x in re.split("/|_", relPathModel) if x not in ["models", "%s.pth.tar"%(name_load_model)]]
    namedParam = [x for x in namedParam if ("=" in x) and x.split("=")[0] and x.split("=")[0]]
    dirParam = {x.split("=")[0]: x.split("=")[1] for x in namedParam}

    for name, val in dirParam.items():
        if name == "alpha":
            dirParam[name] = float(val)

    return dirParam

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelBuffer():
    def __init__(self, listNames: list,  maxBufferSize:int, avg_list:list = None):
        self.num_elem = len(listNames)
        self.maxBufferSize = maxBufferSize
        self.middle_elem_idx = (maxBufferSize // 2)
        self.list_names = listNames
        self.avg_list = avg_list
        self.currBufferLen = 0

        for name in listNames:
            self.__dict__[name] = []

    def update(self, new_elements: dict):
        for k,v in new_elements.items():
            self.__dict__[k].append(v)
            if len(self.__dict__[k]) > self.maxBufferSize:
                self.__dict__[k].pop(0)
            self.currBufferLen = len(self.__dict__[k])


            #if k in self.avg_list:
            #    self.__dict__[k + "_avg"].append(sum(self.__dict__[k]) / len(self.__dict__[k]))
            #    if len(self.__dict__[k + "_avg"]) > self.maxBufferSize:
            #        self.__dict__[k + "_avg"].pop(0)


    def get_middle_elem(self):
        actual_idx = min(self.currBufferLen - 1, self.middle_elem_idx)

        ret = {}
        for name in self.list_names:
            ret[name] = self.__dict__[name][actual_idx]
        for name in self.avg_list:
            ret["mean_" + name] = self.avg_val(name)

        return ret


    def avg_val(self, name:str):
        return sum(self.__dict__[name]) / len(self.__dict__[name])

