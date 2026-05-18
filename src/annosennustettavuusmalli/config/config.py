from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml  # type: ignore
from loguru import logger  # type: ignore

from annosennustettavuusmalli.utils.random_sample_hyperparameters import (
    random_sample_hyperparameters,
)

default_config_file = Path(__file__).parent / "config.yaml"


def load_config(config_file: Path = default_config_file) -> dict:
    """Loads the configuration from the config.yaml file and returns it as a dictionary."""

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


@dataclass
class Hyperparameters:
    gradient_accumulation: int
    patch_size: tuple[int, int, int]
    batch_size: int
    early_stopping_patience: int
    in_channels: int
    out_channels: int
    filters: list[int]
    conv_layers: int
    kernel_size: int
    pool_size: int
    skip_filters: int
    act_func: str
    EPOCHS: int
    warmup_period: int
    warm_restart_every: int
    optimizer: dict
    primary_loss: str
    secondary_loss: str
    experiment_name: str

    @classmethod
    def from_dict(cls, data: dict):
        field_names = {f.name for f in fields(cls)}
        logger.debug(
            f"Creating Hyperparameters from dict. Data keys: {list(data.keys())}, Expected fields: {field_names}"
        )
        return cls(**{k: v for k, v in data.items() if k in field_names})


def get_hyperparameters(
    search_type: str, config: dict = load_config()
) -> list[Hyperparameters] | Hyperparameters:

    if search_type == "default":
        return Hyperparameters.from_dict(config["default"])

    elif search_type == "random_search":
        return [
            Hyperparameters.from_dict(
                random_sample_hyperparameters(config["random_search"])
            )
            for _ in range(config["random_search"]["random_samples"])
        ]

    elif search_type == "manual_search":
        logger.info("Using manual hyperparameter configuration from config.yaml")
        logger.debug(f"Manual hyperparameter config: {config['manual_search']}")
        return Hyperparameters.from_dict(config["manual_search"])

    else:
        raise ValueError(
            f"Unknown search type: {search_type}. Use 'default', 'random_search' or 'manual_search'."
        )


@dataclass
class Configuration:
    file: Path = default_config_file
    structures: dict = field(init=False)
    data_paths: dict = field(init=False)
    train_loader_config: dict = field(init=False)
    default: Hyperparameters = field(init=False)
    random_search: list[Hyperparameters] = field(init=False)
    manual_search: Hyperparameters = field(init=False)

    def __post_init__(self):
        self.config = load_config(self.file)
        self.structures = self.config["structures"]
        self.data_paths = self.config["data_paths"]
        self.train_loader_config = self.config["train_loader_config"]
        self.default = Hyperparameters.from_dict(self.config["default"])
        self.random_search = [
            Hyperparameters.from_dict(
                random_sample_hyperparameters(self.config["random_search"])
            )
            for _ in range(self.config["random_search"]["random_samples"])
        ]
        self.manual_search = Hyperparameters.from_dict(self.config["manual_search"])


if __name__ == "__main__":
    logger.level("DEBUG")

    hyperparams = get_hyperparameters("manual_search")

    if isinstance(hyperparams, list):
        for hp in hyperparams:
            print(hp)
            print(hp.in_channels)
    else:
        print(hyperparams)
        print(hyperparams.in_channels)
