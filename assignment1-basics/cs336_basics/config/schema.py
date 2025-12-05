from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 8
    d_ff: int = 1344  # (8/3) * 512, rounded to multiple of 64
    num_layers: int = 6
    theta: float = 10_000.0
    vocab_size: int = 10_000
    context_length: int = 256


@dataclass
class OptimizerConfig:
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0


@dataclass
class DataConfig:
    train_data_path: str = "data/train.npy"
    val_data_path: str = "data/val.npy"
    batch_size: int = 128


@dataclass
class TrainConfig:
    device: str = "mps"
    dtype: str = "bfloat16"
    max_iters: int = 100_000
    warmup_iters: int = 2_000  # ~1-5% of total
    eval_interval: int = 100
    eval_iters: int = 10
    log_interval: int = 10
    checkpoint_interval: int = 5_000
    checkpoint_dir: str = "checkpoints"
    early_stopping_patience: int = 10  # 0 to disable


@dataclass
class Config:
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    train: TrainConfig


def flatten_config(config: Config) -> dict[str, any]:
    return {f"{section}.{key}": value for section, params in asdict(config).items() for key, value in params.items()}
