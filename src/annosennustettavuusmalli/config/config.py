from dataclasses import dataclass, fields
from pathlib import Path

import yaml  # type: ignore
from loguru import logger  # type: ignore

from annosennustettavuusmalli.utils.random_sample_hyperparameters import (
    random_sample_hyperparameters,
)

default_config_file = Path(__file__).parent / "config.yaml"


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


@dataclass
class Configuration:
    structures: dict
    data_paths: dict
    train_loader_config: dict
    default: dict
    random_search: dict
    manual_search: dict


def load_config(config_file: Path = default_config_file) -> dict:
    """Loads the configuration from the config.yaml file and returns it as a dictionary."""

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


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
