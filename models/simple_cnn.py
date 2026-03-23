from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SimpleCNNConfig:
    """Configuration for SimpleCNN."""

    num_classes: int = 10


class SimpleCNN(nn.Module):
    """Simple CNN baseline for CIFAR-10."""

    def __init__(self, config: SimpleCNNConfig) -> None:
        """Initialize model.

        Args:
            config (SimpleCNNConfig): Model configuration.
        """
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 32 -> 16

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 16 -> 8

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x