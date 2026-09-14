from __future__ import annotations

from dataclasses import dataclass

from .direction import Direction


@dataclass(frozen=True, slots=True)
class Entity:
    x: int
    y: int

    def next_pos(self, direction: Direction, board_size: int) -> Entity:
        dx, dy = direction.delta
        return Entity(
            x=(self.x + dx) % board_size,
            y=(self.y + dy) % board_size,
        )
