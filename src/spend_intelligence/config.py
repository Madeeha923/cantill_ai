from pathlib import Path
import yaml


def load_config(config_path: str | Path) -> dict:
    """
    Load a company's YAML configuration file.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML dictionary.")

    required_keys = ["company", "files"]

    for key in required_keys:
        if key not in config:
            raise ValueError(
                f"Missing required configuration key: {key}"
            )

    return config


def resolve_path(project_root: str | Path, relative_path: str) -> Path:
    """
    Resolve a data file path relative to the project root.
    """

    return Path(project_root) / relative_path
