from __future__ import annotations

from pathlib import Path
from typing import Any

import argbind
import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file into a dictionary suitable for argbind.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a top-level mapping.")
    return data


def parse_args_with_config(config_path: str | Path | None = None):
    """
    Helper to unify CLI arguments and YAML configuration.

    Usage mirrors minicpm-audio:
        args = parse_args_with_config("conf/voxcpm/finetune.yml")
        with argbind.scope(args):
            ...
    """
    cli_args = argbind.parse_args()
    if config_path is None:
        return cli_args

    yaml_args = load_yaml_config(config_path)
    with argbind.scope(cli_args):
        yaml_args = argbind.parse_args(yaml_args=yaml_args, argv=[])
    cli_args.update(yaml_args)
    return cli_args
