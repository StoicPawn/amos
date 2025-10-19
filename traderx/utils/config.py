"""Configuration loader utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ConfigBundle:
    """Simple bundle for primary configuration files."""

    data: Dict[str, Any]
    path: Path

    def get(self, key: str, default: Any | None = None) -> Any:
        """Return a configuration section with optional default."""
        return self.data.get(key, default)


def load_config(path: str | Path) -> ConfigBundle:
    """Load a YAML configuration file."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return ConfigBundle(data=data, path=cfg_path)
