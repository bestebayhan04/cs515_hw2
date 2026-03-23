import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import TrainConfig
from models.mobilenet_cifar import MobileNetCIFAR, MobileNetCIFARConfig
from models.resnet_cifar import BasicBlock, ResNetCIFARConfig, resnet18_cifar
from models.simple_cnn import SimpleCNN, SimpleCNNConfig
from models.transfer_resnet import TransferResNet18, TransferResNet18Config


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(use_cuda: bool) -> torch.device:
    """Get computation device.

    Args:
        use_cuda (bool): Whether to use GPU.

    Returns:
        torch.device: Selected device.
    """
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from logits.

    Args:
        logits (torch.Tensor): Model outputs.
        targets (torch.Tensor): Ground-truth labels.

    Returns:
        float: Accuracy value.
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total


def get_transforms(config: TrainConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create data transforms.

    Args:
        config (TrainConfig): Configuration object.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Train and test transforms.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    if config.resize_to_imagenet:
        train_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return train_transform, test_transform


def get_dataloaders(config: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders.

    Args:
        config (TrainConfig): Configuration object.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test loaders.
    """
    train_transform, test_transform = get_transforms(config)

    train_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    return train_loader, test_loader


def get_model(config: TrainConfig) -> nn.Module:
    """Create model based on configuration.

    Args:
        config (TrainConfig): Configuration object.

    Returns:
        nn.Module: Model instance.
    """
    if config.model_name == "simple_cnn":
        model_config = SimpleCNNConfig(
            num_classes=config.num_classes,
        )
        return SimpleCNN(model_config)

    if config.model_name == "resnet18_cifar":
        model_config = ResNetCIFARConfig(
            block=BasicBlock,
            num_blocks=[2, 2, 2, 2],
            num_classes=config.num_classes,
        )
        return resnet18_cifar(model_config)

    if config.model_name == "transfer_resnet18":
        model_config = TransferResNet18Config(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            resize_to_imagenet=config.resize_to_imagenet,
            freeze_backbone=config.freeze_backbone,
        )
        return TransferResNet18(model_config)

    if config.model_name == "mobilenet_cifar":
        model_config = MobileNetCIFARConfig(
            num_classes=config.num_classes,
            pretrained=config.pretrained,
        )
        return MobileNetCIFAR(model_config)

    raise ValueError(f"Model {config.model_name} is not implemented yet.")


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist.

    Args:
        path (str): Directory path.
    """
    os.makedirs(path, exist_ok=True)