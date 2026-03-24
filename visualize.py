import os
import csv
from typing import Dict, List

import matplotlib.pyplot as plt


def load_history(csv_path: str) -> Dict[str, List[float]]:
    """Load training history from CSV file."""
    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            history["epoch"].append(int(row["epoch"]))
            history["train_loss"].append(float(row["train_loss"]))
            history["train_acc"].append(float(row["train_acc"]))
            history["val_loss"].append(float(row["val_loss"]))
            history["val_acc"].append(float(row["val_acc"]))

    return history


def plot_accuracy_comparison(
    csv_path_1: str,
    label_1: str,
    csv_path_2: str,
    label_2: str,
    output_path: str,
) -> None:
    """Plot train/validation accuracy comparison for two experiments."""
    hist_1 = load_history(csv_path_1)
    hist_2 = load_history(csv_path_2)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_1["epoch"], hist_1["train_acc"], label=f"{label_1} Train")
    plt.plot(hist_1["epoch"], hist_1["val_acc"], label=f"{label_1} Val")
    plt.plot(hist_2["epoch"], hist_2["train_acc"], label=f"{label_2} Train")
    plt.plot(hist_2["epoch"], hist_2["val_acc"], label=f"{label_2} Val")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_loss_comparison(
    csv_path_1: str,
    label_1: str,
    csv_path_2: str,
    label_2: str,
    output_path: str,
) -> None:
    """Plot train/validation loss comparison for two experiments."""
    hist_1 = load_history(csv_path_1)
    hist_2 = load_history(csv_path_2)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_1["epoch"], hist_1["train_loss"], label=f"{label_1} Train")
    plt.plot(hist_1["epoch"], hist_1["val_loss"], label=f"{label_1} Val")
    plt.plot(hist_2["epoch"], hist_2["train_loss"], label=f"{label_2} Train")
    plt.plot(hist_2["epoch"], hist_2["val_loss"], label=f"{label_2} Val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_flops_vs_accuracy(output_path: str) -> None:
    """Plot FLOPs versus test accuracy."""
    model_names = [
        "SimpleCNN",
        "SimpleCNN + KD",
        "ResNet18",
        "Transfer Resize",
        "Transfer CIFAR",
        "MobileNet TGS",
    ]
    flops = [11.25, 11.25, 557.22, 1820.0, 557.78, 26.06]
    test_acc = [80.94, 81.81, 86.33, 80.44, 87.45, 80.27]

    plt.figure(figsize=(8, 5))
    plt.scatter(flops, test_acc)

    for i, name in enumerate(model_names):
        plt.annotate(name, (flops[i], test_acc[i]))

    plt.xlabel("FLOPs / MACs (approx. MMac)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Complexity vs Test Accuracy")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    """Create all required plots."""
    os.makedirs("figures", exist_ok=True)

    # 1) ResNet without LS vs with LS
    plot_accuracy_comparison(
        "./checkpoints/resnet18_baseline/resnet18_cifar_history.csv",
        "ResNet No LS",
        "./checkpoints/resnet18_ls01/resnet18_cifar_history.csv",
        "ResNet LS=0.1",
        "./figures/resnet_label_smoothing_accuracy.png",
    )
    plot_loss_comparison(
        "./checkpoints/resnet18_baseline/resnet18_cifar_history.csv",
        "ResNet No LS",
        "./checkpoints/resnet18_ls01/resnet18_cifar_history.csv",
        "ResNet LS=0.1",
        "./figures/resnet_label_smoothing_loss.png",
    )

    # 2) SimpleCNN baseline vs KD
    plot_accuracy_comparison(
        "./checkpoints/simple_cnn_baseline/simple_cnn_history.csv",
        "SimpleCNN",
        "./checkpoints/simple_cnn_kd/simple_cnn_history.csv",
        "SimpleCNN KD",
        "./figures/simplecnn_kd_accuracy.png",
    )
    plot_loss_comparison(
        "./checkpoints/simple_cnn_baseline/simple_cnn_history.csv",
        "SimpleCNN",
        "./checkpoints/simple_cnn_kd/simple_cnn_history.csv",
        "SimpleCNN KD",
        "./figures/simplecnn_kd_loss.png",
    )

    # 3) Transfer option 1 vs option 2
    plot_accuracy_comparison(
        "./checkpoints/transfer_resize_freeze/transfer_resnet18_history.csv",
        "Resize + Freeze",
        "./checkpoints/transfer_cifar_finetune/transfer_resnet18_history.csv",
        "CIFAR Fine-tune",
        "./figures/transfer_learning_accuracy.png",
    )
    plot_loss_comparison(
        "./checkpoints/transfer_resize_freeze/transfer_resnet18_history.csv",
        "Resize + Freeze",
        "./checkpoints/transfer_cifar_finetune/transfer_resnet18_history.csv",
        "CIFAR Fine-tune",
        "./figures/transfer_learning_loss.png",
    )

    # 4) FLOPs vs Accuracy
    plot_flops_vs_accuracy("./figures/flops_vs_accuracy.png")

    print("All figures saved under ./figures")


if __name__ == "__main__":
    main()