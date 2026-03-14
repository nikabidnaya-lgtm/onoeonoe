from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BotState:
    paused: bool = False
