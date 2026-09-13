from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True, eq=True)
class Entity:
    x: int
    y: int


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

    def move(self, board_size: int, *, grow: bool = False) -> bool:
        dx, dy = self.direction.delta
        next_head = Entity(
            x=(self.head.x + dx) % board_size,
            y=(self.head.y + dy) % board_size,
        )

        body = self.segments if grow else list(self.segments)[:-1]
        if next_head in body:
            return False

        self.segments.appendleft(next_head)
        self.occupied.add(next_head)

        if not grow:
            tail = self.segments.pop()
            self.occupied.remove(tail)

        return True

    def __contains__(self, item: Entity) -> bool:
        return item in self.occupied


class Game:
    epoch: int
    score: int
    snake: Snake
    food: Entity | None

    def __init__(self, size: int):
        self.size = size
        self.epoch = 0
        self.score = 0
        self.snake = Snake(Entity(0, 0), Direction.RIGHT)
        self.food = None

    def _spawn_food(self) -> None:
        pass



def main():
    print("Hello from snake-dqn!")


if __name__ == "__main__":
    main()
