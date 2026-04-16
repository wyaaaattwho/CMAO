from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io_utils import load_json


def load_config(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged

