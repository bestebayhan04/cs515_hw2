from ptflops import get_model_complexity_info

from parameters import TrainConfig
from utils import get_model


def compute_model_flops(model_name: str, image_size: int = 32) -> None:
    """Compute and print FLOPs and parameter count.

    Args:
        model_name (str): Model identifier.
        image_size (int): Input image size.
    """
    config = TrainConfig(
        mode="test",
        model_name=model_name,
        data_dir="./data",
        num_classes=10,
        batch_size=1,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_workers=2,
        image_size=image_size,
        use_cuda=False,
        seed=42,
        save_dir="./checkpoints",
        checkpoint_path="",
        label_smoothing=0.0,
        kd_alpha=0.5,
        kd_temperature=4.0,
        pretrained=False,
        resize_to_imagenet=(image_size == 224),
        teacher_checkpoint_path="",
        distillation=False,
        freeze_backbone=False,
        teacher_guided_smoothing=False,
        val_split=0.1,
    )

    model = get_model(config)

    macs, params = get_model_complexity_info(
        model,
        (3, image_size, image_size),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False,
    )

    print(f"Model: {model_name}")
    print(f"Input size: {image_size}x{image_size}")
    print(f"MACs: {macs}")
    print(f"Params: {params}")


if __name__ == "__main__":
    print("---- SimpleCNN ----")
    compute_model_flops("simple_cnn", 32)

    print("\n---- ResNet18-CIFAR ----")
    compute_model_flops("resnet18_cifar", 32)

    print("\n---- Transfer ResNet18 resized option ----")
    compute_model_flops("transfer_resnet18", 224)

    print("\n---- Transfer ResNet18 CIFAR option ----")
    compute_model_flops("transfer_resnet18", 32)

    print("\n---- MobileNet-CIFAR ----")
    compute_model_flops("mobilenet_cifar", 32)