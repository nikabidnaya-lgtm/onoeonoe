from __future__ import annotations

from typing import Any


def resolve_weights(session_cfg: dict[str, Any]) -> dict[str, float]:
    symbols = session_cfg["symbols"]
    if session_cfg.get("weights"):
        return session_cfg["weights"]
    equal = 1 / len(symbols)
    return {s: equal for s in symbols}
