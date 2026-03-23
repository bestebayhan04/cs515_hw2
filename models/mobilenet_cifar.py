from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


@dataclass
class MobileNetCIFARConfig:
    """Configuration for MobileNetCIFAR."""

    num_classes: int = 10
    pretrained: bool = False


class MobileNetCIFAR(nn.Module):
    """MobileNetV2 adapted for CIFAR-10."""

    def __init__(self, config: MobileNetCIFARConfig) -> None:
        """Initialize model.

        Args:
            config (MobileNetCIFARConfig): Model configuration.
        """
        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT if config.pretrained else None
        self.model = mobilenet_v2(weights=weights)

        # Adapt first convolution for CIFAR-sized input
        first_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=3,
            out_channels=first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Replace classifier for CIFAR-10
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (B, num_classes).
        """
        return self.model(x)