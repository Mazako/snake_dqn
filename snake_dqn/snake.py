from __future__ import annotations

from collections import deque

from .direction import Direction
from .entity import Entity


class Snake:
    def __init__(self, head: Entity, direction: Direction) -> None:
        self.segments: deque[Entity] = deque([head])
        self.occupied: set[Entity] = set(self.segments)
        self.direction = direction

    @property
    def head(self) -> Entity:
        return self.segments[0]

    def change_direction(self, direction: Direction) -> bool:
        if len(self.segments) > 1 and direction.is_opposite_to(self.direction):
            return False

        self.direction = direction
        return True

    def move(
        self, board_size: int, *, grow: bool = False
    ) -> tuple[bool, Entity | None]:
        next_head = self.head.next_pos(self.direction, board_size)
        body = self.segments if grow else list(self.segments)[:-1]
        if next_head in body:
            return False, None

        self.segments.appendleft(next_head)
        self.occupied.add(next_head)

        if not grow:
            tail = self.segments.pop()
            self.occupied.remove(tail)
            return True, tail

        return True, None

    def __contains__(self, item: Entity) -> bool:
        return item in self.occupied
