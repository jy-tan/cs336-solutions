import argparse
import json
from dataclasses import replace
from pathlib import Path

import mlflow

from cs336_basics.config.default import defaultConfig
from cs336_basics.config.schema import Config
from cs336_basics.training.trainer import Trainer

MLFLOW_TRACKING_URI = "http://localhost:5050"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Transformer LM", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(Path(__file__).parent.parent / "tokenized_data/TinyStoriesV2-GPT4-train.npy"),
        help="Path to tokenized training dataset",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=str(Path(__file__).parent.parent / "tokenized_data/TinyStoriesV2-GPT4-valid.npy"),
        help="Path to tokenized validation dataset",
    )
    parser.add_argument("--config-override", type=str, default=None, help="Configs to override (JSON string)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


def apply_overrides(config: Config, overrides: dict) -> Config:
    """
    Apply nested overrides to config.

    Example override JSON:
    {
        "model": {"d_model": 256, "num_layers": 4},
        "optimizer": {"max_lr": 1e-3},
        "data": {"batch_size": 32},
        "train": {"max_iters": 50000}
    }
    """
    new_config = config

    if "model" in overrides:
        new_config = replace(new_config, model=replace(config.model, **overrides["model"]))
    if "optimizer" in overrides:
        new_config = replace(new_config, optimizer=replace(config.optimizer, **overrides["optimizer"]))
    if "data" in overrides:
        new_config = replace(new_config, data=replace(config.data, **overrides["data"]))
    if "train" in overrides:
        new_config = replace(new_config, train=replace(config.train, **overrides["train"]))

    return new_config


def main():
    args = parse_args()
    config = defaultConfig

    config = replace(config, data=replace(config.data, train_data_path=args.train_path, val_data_path=args.val_path))

    if args.config_override:
        overrides = json.loads(args.config_override)
        config = apply_overrides(config, overrides)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    trainer = Trainer(config)

    if args.resume:
        trainer.load(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
