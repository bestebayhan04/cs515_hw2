import torch
import torch.nn as nn

from parameters import TrainConfig
from utils import get_dataloaders, get_device, get_model


@torch.no_grad()
def test_model(config: TrainConfig) -> None:
    """Evaluate model on test set.

    Args:
        config (TrainConfig): Configuration object.
    """
    device = get_device(config.use_cuda)

    _, test_loader = get_dataloaders(config)

    model = get_model(config).to(device)

    if not config.checkpoint_path:
        raise ValueError("Please provide --checkpoint_path for test mode.")

    # Load checkpoint safely
    state_dict = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("No samples found in test loader.")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_acc * 100:.2f}%")