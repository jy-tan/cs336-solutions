import os
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from cs336_basics.config.schema import Config, flatten_config
from cs336_basics.training.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.training.data import get_batch
from cs336_basics.training.loss import cross_entropy_loss
from cs336_basics.training.optimizer import AdamW, clip_gradients, cosine_lr_schedule
from cs336_basics.transformer.transformer import Transformer


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = config.train.device
        self.dtype = getattr(torch, config.train.dtype)

        self.train_data = np.load(config.data.train_data_path, mmap_mode="r")
        print(f"Train data: {len(self.train_data)} tokens")
        self.val_data = np.load(config.data.val_data_path, mmap_mode="r")
        print(f"Validation data: {len(self.val_data)} tokens")

        self.model = Transformer(
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
            theta=config.model.theta,
            vocab_size=config.model.vocab_size,
            context_length=config.model.context_length,
            num_layers=config.model.num_layers,
            device=self.device,
            dtype=self.dtype,
        )

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")

        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=config.optimizer.max_lr,
            betas=(config.optimizer.beta1, config.optimizer.beta2),
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
        )

        Path(config.train.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self.use_mlflow = True

        self.iter_num = 0

    def get_lr(self) -> float:
        """Get learning rate for current iteration using cosine schedule."""
        return cosine_lr_schedule(
            self.iter_num,
            max_learning_rate=self.config.optimizer.max_lr,
            min_learning_rate=self.config.optimizer.min_lr,
            warmup_iters=self.config.train.warmup_iters,
            cosine_cycle_iters=self.config.train.max_iters,
        )

    def set_lr(self, lr: float):
        """Update learning rate in optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        self.model.eval()
        losses = {}

        for split, data in [("train", self.train_data), ("val", self.val_data)]:
            total_loss = 0.0

            for _ in range(self.config.train.eval_iters):
                inputs, targets = get_batch(
                    data, self.config.data.batch_size, self.config.model.context_length, self.device
                )
                logits = self.model(inputs)
                loss = cross_entropy_loss(logits, targets)
                total_loss += loss.item()

            losses[split] = total_loss / self.config.train.eval_iters

        self.model.train()

        return losses

    def save(self, path: str | None) -> None:
        if path is None:
            path = os.path.join(self.config.train.checkpoint_dir, f"checkpoint_{self.iter_num}.pt")

        save_checkpoint(self.model, self.optimizer, self.iter_num, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        self.iter_num = load_checkpoint(path, self.model, self.optimizer)
        print(f"Loaded checkpoint from {path} at iteration {self.iter_num}")

    def train(self):
        print(f"Starting training, device={self.device}, max_iters={self.config.train.max_iters}")

        # Start MLflow run
        if self.use_mlflow:
            mlflow.set_experiment("cs336-basics")
            mlflow.start_run(log_system_metrics=True)
            mlflow.log_params(flatten_config(self.config))

        self.model.train()
        t0 = time.time()

        while self.iter_num < self.config.train.max_iters:
            lr = self.get_lr()
            self.set_lr(lr)

            inputs, targets = get_batch(
                self.train_data,
                batch_size=self.config.data.batch_size,
                context_length=self.config.model.context_length,
                device=self.device,
            )

            logits = self.model(inputs)
            loss = cross_entropy_loss(logits, targets)

            # Optimizer is not part of computation graph.
            # In the optimizer implementation, we deliberately used in-place operations on param.data.
            self.optimizer.zero_grad()

            # Backward pass - computes gradients.
            # Store gradients in param.grad
            loss.backward()

            grad_norm = clip_gradients(self.model.parameters(), self.config.optimizer.grad_clip)

            # Update weights (param.data) based on gradients (in param.grad)
            self.optimizer.step()

            # Logging
            if self.iter_num % self.config.train.log_interval == 0:
                t1 = time.time()
                duration = t1 - t0
                tokens_per_sec = (
                    self.config.data.batch_size
                    * self.config.model.context_length
                    * self.config.train.log_interval
                    / duration
                )
                print(
                    f"iter {self.iter_num:6d} | loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                    f"{tokens_per_sec:.0f} tok/s"
                )

                if self.use_mlflow:
                    mlflow.log_metrics(
                        {
                            "train/loss": loss.item(),
                            "train/lr": lr,
                            "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                            "train/tokens_per_sec": tokens_per_sec,
                        },
                        step=self.iter_num,
                    )

                t0 = time.time()

            # Eval
            if self.iter_num > 0 and self.iter_num % self.config.train.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"iter {self.iter_num} | train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f}")

                if self.use_mlflow:
                    mlflow.log_metrics(
                        {
                            "eval/train_loss": losses["train"],
                            "eval/val_loss": losses["val"],
                        },
                        step=self.iter_num,
                    )

            # Checkpointing
            if self.iter_num > 0 and self.iter_num % self.config.train.checkpoint_interval == 0:
                self.save()

            self.iter_num += 1

        self.save(os.path.join(self.config.train.checkpoint_dir, "checkpoint_final.pt"))

        if self.use_mlflow:
            mlflow.log_artifact(os.path.join(self.config.train.checkpoint_dir, "checkpoint_final.pt"))
            mlflow.end_run()

        print("Training complete.")
