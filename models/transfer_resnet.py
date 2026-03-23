from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


@dataclass
class TransferResNet18Config:
    """Configuration for TransferResNet18."""

    num_classes: int = 10
    pretrained: bool = True
    resize_to_imagenet: bool = True
    freeze_backbone: bool = False


class TransferResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10 with transfer learning options."""

    def __init__(self, config: TransferResNet18Config) -> None:
        """Initialize model.

        Args:
            config (TransferResNet18Config): Model configuration.
        """
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if config.pretrained else None
        self.model = resnet18(weights=weights)

        # Adapt for CIFAR (no resizing to 224x224)
        if not config.resize_to_imagenet:
            self.model.conv1 = nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.model.maxpool = nn.Identity()

        # Replace final classifier
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, config.num_classes)

        # Freeze backbone if required
        if config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze classifier
            for param in self.model.fc.parameters():
                param.requires_grad = True

            # If CIFAR adaptation is used, also unfreeze first conv
            if not config.resize_to_imagenet:
                for param in self.model.conv1.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        return self.model(x)