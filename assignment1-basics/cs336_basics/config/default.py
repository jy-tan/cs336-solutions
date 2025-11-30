from cs336_basics.config.schema import Config, DataConfig, ModelConfig, OptimizerConfig, TrainConfig

defaultConfig = Config(
    model=ModelConfig(),
    optimizer=OptimizerConfig(),
    data=DataConfig(),
    train=TrainConfig(),
)
