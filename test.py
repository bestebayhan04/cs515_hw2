import torch
import torch.nn as nn

from parameters import TrainConfig
from utils import get_dataloaders, get_device, get_model


@torch.no_grad()
def test_model(config: TrainConfig) -> None:
    device = get_device(config.use_cuda)
    _, test_loader = get_dataloaders(config)

    model = get_model(config).to(device)
    if not config.checkpoint_path:
        raise ValueError("Please provide --checkpoint_path for test mode.")

    model.load_state_dict(torch.load(config.checkpoint_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_acc * 100:.2f}%")