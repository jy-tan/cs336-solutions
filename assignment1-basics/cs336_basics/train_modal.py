import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "einops", "jaxtyping", "tqdm", "wandb")
    .add_local_dir("cs336_basics", remote_path="/root/cs336_basics")
)

app = modal.App("cs336-training", image=image)

# This should already be created beforehand, data uploaded via CLI
volume = modal.Volume.from_name("cs336-data", create_if_missing=True)


@app.function(
    gpu="A10G",
    timeout=4 * 60 * 60,  # 4 hours max
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config_override: dict | None = None):
    import os
    import sys

    sys.path.insert(0, "/root")

    # Verify data exists
    train_path = "/data/TinyStoriesV2-GPT4-train.npy"
    val_path = "/data/TinyStoriesV2-GPT4-valid.npy"

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Training data not found at {train_path}. Upload with: modal volume put cs336-data ..."
        )
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at {val_path}")

    from dataclasses import replace

    import wandb

    from cs336_basics.config.schema import flatten_config
    from cs336_basics.config.small import smallConfig
    from cs336_basics.training.trainer import Trainer

    config = smallConfig

    # Override paths to use volume
    config = replace(
        config,
        data=replace(
            config.data,
            train_data_path=train_path,
            val_data_path=val_path,
        ),
        train=replace(
            config.train,
            device="cuda",
            checkpoint_dir="/data/checkpoints",
        ),
    )

    if config_override:
        from cs336_basics.train import apply_overrides

        config = apply_overrides(config, config_override)

    wandb.init(project="cs336-basics", config=flatten_config(config))

    print(f"Training config: {config}")

    trainer = Trainer(config)
    trainer.wandb_run = wandb.run
    trainer.train()

    wandb.finish()
    volume.commit()

    print("Training complete! Checkpoints saved to volume.")


@app.local_entrypoint()
def main():
    train.remote(
        config_override={
            # "model": {"d_model": 768, "num_layers": 12},
            # "optimizer": {"max_lr": 3e-4},
        }
    )
