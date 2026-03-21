import os
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for images, targets in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)
        total_batches += 1

    return {
        "loss": running_loss / total_batches,
        "acc": running_acc / total_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    total_batches = 0

    for images, targets in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)
        total_batches += 1

    return {
        "loss": running_loss / total_batches,
        "acc": running_acc / total_batches,
    }


def train_model(config: TrainConfig) -> None:
    device = get_device(config.use_cuda)
    train_loader, test_loader = get_dataloaders(config)
    model = get_model(config).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    ensure_dir(config.save_dir)
    best_acc = 0.0

    for epoch in range(config.epochs):
        print(f"\nEpoch [{epoch + 1}/{config.epochs}]")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
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

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            checkpoint_file = os.path.join(config.save_dir, f"{config.model_name}_best.pth")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"Saved best model to: {checkpoint_file}")

    print(f"\nBest Validation Accuracy: {best_acc * 100:.2f}%")