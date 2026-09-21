from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class GameState:
    direction: tuple[float, float, float, float]
    blocked_left: float
    blocked_forward: float
    blocked_right: float
    dx: float
    dy: float
    occupied_ratio: float
    tail_dx: float
    tail_dy: float

    def features(self) -> torch.Tensor:
        return torch.tensor(
            (
                *self.direction,
                self.blocked_left,
                self.blocked_forward,
                self.blocked_right,
                self.dx,
                self.dy,
                self.occupied_ratio,
                self.tail_dx,
                self.tail_dy,
            ),
            dtype=torch.float32,
        )
