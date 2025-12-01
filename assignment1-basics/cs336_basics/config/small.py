from cs336_basics.config.schema import Config, DataConfig, ModelConfig, OptimizerConfig, TrainConfig

smallConfig = Config(
    model=ModelConfig(num_heads=16, num_layers=4),
    optimizer=OptimizerConfig(max_lr=1e-4, min_lr=1e-5, weight_decay=0.3),
    data=DataConfig(),
    train=TrainConfig(),
)
