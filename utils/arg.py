

import argparse
import torch.nn as nn

def is_prunable_module(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        return True
    return False

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')


    parser.add_argument('--outdir', type=str, default="./run", help='path to output dir')
    parser.add_argument('--override_dir', action='store_true', help='set if folder can be overritten')
    parser.add_argument('--manual_seed', type=int, default=None, help='manual seed')

    '''
    pruner
    '''
    parser.add_argument('--prune_rate', type=float, default=0.9, help='fraction of pruned weights')
    parser.add_argument('--eta', type=float, default=1.05, help='fraction of pruned weights')
    parser.add_argument('--lambda_init', type=float, default=5e-6, help='fraction of pruned weights')
    parser.add_argument('--size_model_buffer', type=int, default=1, help='averaging buffer for model_acc in ART')

    '''
    setting dataset
    '''
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['cifar10', 'cifar100', 'tinyimagenet'], help='name dataset')
    parser.add_argument('--path_data', type=str, default="./data", help='path to dataset')

    '''
    setting models
    '''
    parser.add_argument('--model_arch', type=str, default="resnet", choices=['resnet', 'vgg'], help='architecture models')
    parser.add_argument('--model_depth', type=int, default="32", help='architecture depth value')
    parser.add_argument('--path_load_model', type=str, default=None, help='path to models-file')

    '''
    settings training
    '''
    parser.add_argument('--epochs', type=int, default=160, help='number of epochs for fine_tuning')
    parser.add_argument('--warmup_epochs', type=int, default=60, help='number of epochs to warmup')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--regularization_func', type=str, choices=['HS', 'L1', "L2"], default="HS")

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--lr_step', type=int, nargs='+', default=[80, 120])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)


    args = parser.parse_args()
    return args


