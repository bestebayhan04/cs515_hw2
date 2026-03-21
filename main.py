from parameters import get_config
from train import train_model
from test import test_model
from utils import set_seed


def main() -> None:
    config = get_config()
    set_seed(config.seed)

    if config.mode == "train":
        train_model(config)
    elif config.mode == "test":
        test_model(config)
    else:
        raise ValueError(f"Unsupported mode: {config.mode}")


if __name__ == "__main__":
    main()