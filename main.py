from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from random import Random

from random_set import RandomSet


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


@dataclass(frozen=True, slots=True, eq=True)
class Entity:
    x: int
    y: int

    def next_pos(self, direction: Direction, board_size: int):
        dx, dy = direction.delta
        return Entity(
            x=(self.x + dx) % board_size,
            y=(self.y + dy) % board_size,
        )


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
        dx, dy = self.direction.delta
        next_head = Entity(
            x=(self.head.x + dx) % board_size,
            y=(self.head.y + dy) % board_size,
        )

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


class Game:
    epoch: int
    score: int
    snake: Snake
    food: Entity

    def __init__(self, size: int):
        self.size = size
        self.score = 0
        self.epoch = 0
        self.snake = Snake(Entity(0, 0), Direction.RIGHT)
        self.rng = Random()
        self.free_coords_pool = RandomSet(
            {Entity(x, y) for x in range(size) for y in range(size)}
        )
        for segment in self.snake.occupied:
            self.free_coords_pool.discard(segment)
        self.food = self.free_coords_pool.pop_random(self.rng)

    def next_food(self):
        self.food = self.free_coords_pool.pop_random(self.rng)

    def change_direction(self, direction: Direction) -> bool:
        return self.snake.change_direction(direction)

    def tick(self) -> bool:
        self.epoch += 1
        next_head = self.snake.head.next_pos(self.snake.direction, self.size)
        will_eat = next_head == self.food
        moved, tail = self.snake.move(self.size, grow=will_eat)
        if not moved:
            return False

        if tail is not None:
            self.free_coords_pool.add(tail)
        self.free_coords_pool.discard(next_head)

        if will_eat:
            self.score += 1
            self.next_food()

        return True


def main():
    print("Hello from snake-dqn!")


if __name__ == "__main__":
    main()
