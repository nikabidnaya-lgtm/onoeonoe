from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    raw: dict[str, Any]

    @property
    def symbols(self) -> list[str]:
        return self.raw["trading"]["symbols"]


def _expand_env(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
        return os.getenv(node[2:-1], "")
    return node


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    load_dotenv()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(raw=_expand_env(data))
