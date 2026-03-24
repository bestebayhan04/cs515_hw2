import os
from typing import Any, Dict, Optional
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from losses import (
    DistillationLoss,
    DistillationLossConfig,
    TeacherGuidedLabelSmoothingLoss,
)
from parameters import TrainConfig
from utils import (
    accuracy_from_logits,
    ensure_dir,
    get_dataloaders,
    get_device,
    get_model,
)


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Train model for one epoch.

    Args:
        model (nn.Module): Model to train.
        loader (Any): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): Device to use.
        teacher_model (Optional[nn.Module]): Teacher model for distillation.

    Returns:
        Dict[str, float]: Average loss and accuracy.
    """
    model.train()

    running_loss: float = 0.0
    running_acc: float = 0.0
    total_batches: int = 0

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        student_logits = model(images)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            loss = criterion(student_logits, teacher_logits, targets)
        else:
            loss = criterion(student_logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(student_logits, targets)
        total_batches += 1

    if total_batches == 0:
        raise ValueError("No training batches found.")

    return {
        "loss": running_loss / total_batches,
        "acc": running_acc / total_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation or test data.

    Args:
        model (nn.Module): Model to evaluate.
        loader (Any): Validation or test data loader.
        device (torch.device): Device to use.

    Returns:
        Dict[str, float]: Average loss and accuracy.
    """
    model.eval()

    running_loss: float = 0.0
    running_acc: float = 0.0
    total_batches: int = 0

    criterion = nn.CrossEntropyLoss()

    for images, targets in tqdm(loader, desc="Evaluation", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)
        total_batches += 1

    if total_batches == 0:
        raise ValueError("No evaluation batches found.")

    return {
        "loss": running_loss / total_batches,
        "acc": running_acc / total_batches,
    }


def build_teacher_config(config: TrainConfig) -> TrainConfig:
    """Create teacher configuration.

    Args:
        config (TrainConfig): Student training configuration.

    Returns:
        TrainConfig: Teacher configuration.
    """
    return TrainConfig(
        mode="test",
        model_name="resnet18_cifar",
        data_dir=config.data_dir,
        num_classes=config.num_classes,
        batch_size=config.batch_size,
        epochs=1,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_workers=config.num_workers,
        image_size=32,
        use_cuda=config.use_cuda,
        seed=config.seed,
        save_dir=config.save_dir,
        checkpoint_path=config.teacher_checkpoint_path,
        label_smoothing=0.0,
        kd_alpha=config.kd_alpha,
        kd_temperature=config.kd_temperature,
        pretrained=False,
        resize_to_imagenet=False,
        teacher_checkpoint_path="",
        distillation=False,
        freeze_backbone=False,
        teacher_guided_smoothing=False,
        val_split=config.val_split,
    )


def train_model(config: TrainConfig) -> None:
    """Train model using the given configuration.

    Args:
        config (TrainConfig): Configuration object.
    """
    device = get_device(config.use_cuda)
    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)

    teacher_model: Optional[nn.Module] = None

    if config.distillation or config.teacher_guided_smoothing:
        if not config.teacher_checkpoint_path:
            raise ValueError("Please provide --teacher_checkpoint_path.")

        teacher_config = build_teacher_config(config)
        teacher_model = get_model(teacher_config).to(device)

        teacher_state_dict = torch.load(
            config.teacher_checkpoint_path,
            map_location=device,
        )
        teacher_model.load_state_dict(teacher_state_dict)
        teacher_model.eval()

        if config.teacher_guided_smoothing:
            criterion = TeacherGuidedLabelSmoothingLoss()
        else:
            loss_config = DistillationLossConfig(
                alpha=config.kd_alpha,
                temperature=config.kd_temperature,
            )
            criterion = DistillationLoss(loss_config)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    optimizer = optim.Adam(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    ensure_dir(config.save_dir)
    best_acc: float = -1.0
    history: list[dict[str, float]] = []
    best_checkpoint_path: Optional[str] = None

    for epoch in range(config.epochs):
        print(f"\nEpoch [{epoch + 1}/{config.epochs}]")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            teacher_model=teacher_model,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['acc'] * 100:.2f}%"
        )
        print(
            f"Val Loss:   {val_metrics['loss']:.4f} | "
            f"Val Acc:   {val_metrics['acc'] * 100:.2f}%"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"] * 100,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"] * 100,
            }
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]

            if config.teacher_guided_smoothing:
                suffix = "_tgs_best.pth"
            elif config.distillation:
                suffix = "_kd_best.pth"
            else:
                suffix = "_best.pth"

            best_checkpoint_path = os.path.join(
                config.save_dir,
                f"{config.model_name}{suffix}",
            )
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved best model to: {best_checkpoint_path}")

    print(f"\nBest Validation Accuracy: {best_acc * 100:.2f}%")

    if best_checkpoint_path is None:
        raise ValueError("Best checkpoint was not saved during training.")

    best_state_dict = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_state_dict)

    test_metrics = evaluate(
        model=model,
        loader=test_loader,
        device=device,
    )

    print(
        f"Final Test Loss: {test_metrics['loss']:.4f} | "
        f"Final Test Acc: {test_metrics['acc'] * 100:.2f}%"
    )

    history_path = os.path.join(config.save_dir, f"{config.model_name}_history.csv")

    with open(history_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)

    print(f"Saved training history to: {history_path}")

    print("\n=== FINAL RESULTS ===")
    print(f"Last Train Acc: {train_metrics['acc'] * 100:.2f}%")
    print(f"Best Val Acc:   {best_acc * 100:.2f}%")
    print(f"Test Acc:       {test_metrics['acc'] * 100:.2f}%")