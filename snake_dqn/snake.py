from __future__ import annotations

from collections import deque

from .direction import Direction
from .position import Position


class Snake:
    def __init__(self, head: Position, direction: Direction) -> None:
        self.segments: deque[Position] = deque([head])
        self.occupied: set[Position] = set(self.segments)
        self.direction = direction

    @property
    def head(self) -> Position:
        return self.segments[0]

    def change_direction(self, direction: Direction) -> bool:
        if len(self.segments) > 1 and direction.is_opposite_to(self.direction):
            return False

        self.direction = direction
        return True

    def next_head(self, direction: Direction, board_size: int) -> Position:
        return self.head.next_pos(direction, board_size)

    def would_collide(self, next_head: Position, *, grow: bool) -> bool:
        tail = self.segments[-1]
        return next_head in self and (grow or next_head != tail)

    def advance(self, next_head: Position, *, grow: bool) -> Position | None:
        self.segments.appendleft(next_head)
        self.occupied.add(next_head)

        if not grow:
            tail = self.segments.pop()
            if tail != next_head:
                self.occupied.remove(tail)
            return tail

        return None

    def __contains__(self, item: Position) -> bool:
        return item in self.occupied

    def __len__(self) -> int:
        return len(self.segments)
