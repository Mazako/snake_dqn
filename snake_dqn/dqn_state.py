from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GameState:
    direction: tuple[float, float, float, float]
    blocked_left: float
    blocked_forward: float
    blocked_right: float
    dx: float
    dy: float
