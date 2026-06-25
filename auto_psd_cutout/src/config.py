from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config and normalize paths relative to the project root."""
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    root = config_path.parent
    config["project_root"] = root
    config["input_dir"] = _resolve_path(root, config.get("input_dir", "input"))
    config["output_dir"] = _resolve_path(root, config.get("output_dir", "output"))
    return config


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()
