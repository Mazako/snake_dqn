from __future__ import annotations

from enum import Enum


class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

    @property
    def delta(self) -> tuple[int, int]:
        return self.value

    def is_opposite_to(self, other: Direction) -> bool:
        dx, dy = self.delta
        other_dx, other_dy = other.delta
        return (dx, dy) == (-other_dx, -other_dy)
