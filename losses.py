from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DistillationLossConfig:
    """Configuration for DistillationLoss."""

    alpha: float = 0.5
    temperature: float = 4.0


class DistillationLoss(nn.Module):
    """Knowledge distillation loss (CE + KL)."""

    def __init__(self, config: DistillationLossConfig) -> None:
        """Initialize loss.

        Args:
            config (DistillationLossConfig): Loss configuration.
        """
        super().__init__()
        self.alpha = config.alpha
        self.temperature = config.temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
            student_logits (torch.Tensor): Student outputs.
            teacher_logits (torch.Tensor): Teacher outputs.
            targets (torch.Tensor): Ground-truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        # Hard loss (ground truth)
        hard_loss = self.ce_loss(student_logits, targets)

        # Soft loss (teacher guidance)
        t = self.temperature
        student_log_probs = F.log_softmax(student_logits / t, dim=1)
        teacher_probs = F.softmax(teacher_logits / t, dim=1)

        soft_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (t ** 2)

        return self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss


class TeacherGuidedLabelSmoothingLoss(nn.Module):
    """Label smoothing guided by teacher confidence."""

    def __init__(self) -> None:
        """Initialize loss."""
        super().__init__()

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute teacher-guided smoothing loss.

        Args:
            student_logits (torch.Tensor): Student outputs.
            teacher_logits (torch.Tensor): Teacher outputs.
            targets (torch.Tensor): Ground-truth labels.

        Returns:
            torch.Tensor: Loss value.
        """
        num_classes = student_logits.size(1)
        batch_size = targets.size(0)

        with torch.no_grad():
            teacher_probs = F.softmax(teacher_logits, dim=1)

            soft_targets = torch.zeros_like(student_logits)

            true_class_probs = teacher_probs[torch.arange(batch_size), targets]
            remaining = 1.0 - true_class_probs
            other_prob = remaining / (num_classes - 1)

            soft_targets.fill_(0.0)
            soft_targets += other_prob.unsqueeze(1)

            soft_targets[torch.arange(batch_size), targets] = true_class_probs

        student_log_probs = F.log_softmax(student_logits, dim=1)
        loss = -(soft_targets * student_log_probs).sum(dim=1).mean()

        return loss