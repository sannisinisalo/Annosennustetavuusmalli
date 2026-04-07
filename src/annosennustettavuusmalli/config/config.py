from pathlib import Path

import yaml  # type: ignore

config_file = Path(__file__).parent / "config.yaml"


def load_default_config() -> dict:
    """Loads the configuration from the config.yaml file and returns it as a dictionary."""

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
