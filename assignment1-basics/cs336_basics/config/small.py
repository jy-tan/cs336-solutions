from cs336_basics.config.schema import Config, DataConfig, ModelConfig, OptimizerConfig, TrainConfig

smallConfig = Config(
    model=ModelConfig(num_heads=16, num_layers=4),
    optimizer=OptimizerConfig(
        max_lr=6e-4,
        min_lr=6e-5,  # 10% of max lr
        weight_decay=0.1,
    ),
    data=DataConfig(),
    train=TrainConfig(
        max_iters=10_000,
        warmup_iters=500,
        eval_interval=100,
        early_stopping_patience=0,  # Disable for hyperparameter search
    ),
)
