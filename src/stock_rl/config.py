from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def project_path(config_path: str | Path, *parts: str) -> Path:
    root = Path(config_path).resolve().parents[1]
    return root.joinpath(*parts)
