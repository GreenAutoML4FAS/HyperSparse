
import torch
import torchvision
import torchvision.transforms as transforms

from dataclasses import dataclass
from typing import Tuple



@dataclass
class DatasetConstants:
    dataset_size: int
    num_classes: int
    crop_size: int
    channel_means: Tuple[float]
    channel_stds: Tuple[float]


@dataclass
class TinyImageNetConstants(DatasetConstants):
    dataset_size: int = 0
    num_classes: int = 200
    crop_size: int = 64
    channel_means: Tuple[float] = (0.48024578664982126, 0.44807218089384643, 0.3975477478649648)
    channel_stds: Tuple[float] = (0.2769864069088257, 0.26906448510256, 0.282081906210584)


@dataclass
class Cifar10Constants(DatasetConstants):
    dataset_size: int = 50000
    num_classes: int = 10
    crop_size: int = 32
    channel_means: Tuple[float] = (0.4914, 0.4822, 0.4465)
    channel_stds: Tuple[float] = (0.2023, 0.1994, 0.2010)


@dataclass
class Cifar100Constants(DatasetConstants):
    dataset_size: int = 50000
    num_classes: int = 100
    crop_size: int = 32
    channel_means: Tuple[float] = (0.5071, 0.4867, 0.4408)
    channel_stds: Tuple[float] = (0.2675, 0.2565, 0.2761)


def get_dataset_constants(dataset: str):
    constants_dict = {
        "tinyimagenet": TinyImageNetConstants,
        "cifar10": Cifar10Constants,
        "cifar100": Cifar100Constants,
    }

    if dataset.lower() not in constants_dict:
        raise ValueError(f"Invalid dataset name: {dataset}")
    return constants_dict[dataset.lower()]()


def get_train_transforms(args):
    constants = get_dataset_constants(args.dataset)

    trans = [transforms.RandomCrop(constants.crop_size, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(constants.channel_means, constants.channel_stds),
             ]

    return transforms.Compose(trans)


def get_val_transforms(args):
    constants = get_dataset_constants(args.dataset)

    trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=constants.channel_means, std=constants.channel_stds),
    ]
    return transforms.Compose(trans)


def get_train_loader(args):

    train_tf = get_train_transforms(args)

    if args.dataset in ["cifar10", "cifar100"]:
        ds_string = (args.dataset).upper()
        dataset = torchvision.datasets.__dict__[ds_string](args.path_data, train=True, download=True, transform=train_tf)
    elif args.dataset in ["tinyimagenet"]:
        dataset = torchvision.datasets.ImageFolder(args.path_data + "/train", transform=train_tf)
    else:
        assert False

    train_loader = torch.utils.data.DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=args.workers)

    return train_loader


def get_test_loader(args):
    eval_tf = get_val_transforms(args)

    if args.dataset in ["cifar10", "cifar100"]:
        ds_string = (args.dataset).upper()
        dataset = torchvision.datasets.__dict__[ds_string](args.path_data, train=False, download=False, transform=eval_tf)
    elif args.dataset in ["tinyimagenet"]:
        dataset = torchvision.datasets.ImageFolder(args.path_data + "/val", eval_tf)
    else:
        assert False

    val_loader = torch.utils.data.DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.workers)

    return val_loader

def get_dataloader(args):
    constants_dataset = get_dataset_constants(args.dataset)

    train_loader = get_train_loader(args)
    test_loader = get_test_loader(args)

    return train_loader, test_loader, constants_dataset

