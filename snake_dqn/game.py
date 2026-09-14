from __future__ import annotations

from random import Random

from .direction import Direction
from .entity import Entity
from .random_set import RandomSet
from .snake import Snake


class Game:
    epoch: int
    score: int
    snake: Snake
    food: Entity

    def __init__(self, size: int) -> None:
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

    def next_food(self) -> None:
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
