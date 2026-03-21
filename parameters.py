from dataclasses import dataclass
import argparse


@dataclass
class TrainConfig:
    mode: str
    model_name: str
    data_dir: str
    num_classes: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    image_size: int
    use_cuda: bool
    seed: int
    save_dir: str
    checkpoint_path: str
    label_smoothing: float
    kd_alpha: float
    kd_temperature: float
    pretrained: bool
    resize_to_imagenet: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transfer Learning and Knowledge Distillation on CIFAR-10")

    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument(
        "--model_name",
        type=str,
        default="simple_cnn",
        choices=[
            "simple_cnn",
            "resnet18_cifar",
            "mobilenet_cifar",
            "transfer_resnet18",
            "transfer_vgg16",
        ],
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--use_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resize_to_imagenet", action="store_true")

    return parser


def get_config() -> TrainConfig:
    parser = build_parser()
    args = parser.parse_args()

    image_size = 224 if args.resize_to_imagenet else args.image_size

    return TrainConfig(
        mode=args.mode,
        model_name=args.model_name,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=image_size,
        use_cuda=args.use_cuda,
        seed=args.seed,
        save_dir=args.save_dir,
        checkpoint_path=args.checkpoint_path,
        label_smoothing=args.label_smoothing,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        pretrained=args.pretrained,
        resize_to_imagenet=args.resize_to_imagenet,
    )