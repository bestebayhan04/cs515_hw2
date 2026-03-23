from dataclasses import dataclass, field
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BasicBlockConfig:
    """Configuration for a basic residual block."""

    in_channels: int
    out_channels: int
    stride: int = 1


@dataclass
class ResNetCIFARConfig:
    """Configuration for ResNetCIFAR."""

    block: Type["BasicBlock"]
    num_blocks: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_classes: int = 10


class BasicBlock(nn.Module):
    """Basic residual block for CIFAR ResNet."""

    expansion: int = 1

    def __init__(self, config: BasicBlockConfig) -> None:
        """Initialize block.

        Args:
            config (BasicBlockConfig): Block configuration.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=3,
            stride=config.stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(config.out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=config.out_channels,
            out_channels=config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(config.out_channels)

        self.shortcut = nn.Sequential()
        if config.stride != 1 or config.in_channels != config.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    kernel_size=1,
                    stride=config.stride,
                    bias=False,
                ),
                nn.BatchNorm2d(config.out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class ResNetCIFAR(nn.Module):
    """ResNet adapted for CIFAR-10."""

    def __init__(self, config: ResNetCIFARConfig) -> None:
        """Initialize model.

        Args:
            config (ResNetCIFARConfig): Model configuration.
        """
        super().__init__()

        self.block = config.block
        self.num_blocks = config.num_blocks
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(out_channels=64, num_blocks=self.num_blocks[0], stride=1)
        self.layer2 = self._make_layer(out_channels=128, num_blocks=self.num_blocks[1], stride=2)
        self.layer3 = self._make_layer(out_channels=256, num_blocks=self.num_blocks[2], stride=2)
        self.layer4 = self._make_layer(out_channels=512, num_blocks=self.num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.block.expansion, config.num_classes)

    def _make_layer(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a ResNet layer.

        Args:
            out_channels (int): Output channels.
            num_blocks (int): Number of blocks.
            stride (int): First block stride.

        Returns:
            nn.Sequential: Layer module.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers: List[nn.Module] = []

        for current_stride in strides:
            block_config = BasicBlockConfig(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=current_stride,
            )
            layers.append(self.block(block_config))
            self.in_channels = out_channels * self.block.expansion

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_cifar(config: ResNetCIFARConfig | None = None) -> ResNetCIFAR:
    """Build ResNet-18 for CIFAR.

    Args:
        config (ResNetCIFARConfig | None): Model configuration. If None,
            a default ResNet-18 CIFAR configuration is used.

    Returns:
        ResNetCIFAR: Model instance.
    """
    if config is None:
        config = ResNetCIFARConfig(block=BasicBlock)

    return ResNetCIFAR(config)