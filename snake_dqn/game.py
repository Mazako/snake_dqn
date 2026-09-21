from __future__ import annotations

from random import Random

import numpy as np

from snake_dqn.agent import RelativeAction

from .direction import Direction
from .dqn_state import GameState
from .position import Position
from .random_set import RandomSet
from .snake import Snake


class Game:
    epoch: int
    score: int
    snake: Snake
    food: Position

    def __init__(self, size: int) -> None:
        self.size = size
        self.score = 0
        self.epoch = 0
        self.snake = Snake(Position(0, 0), Direction.RIGHT)
        self.rng = Random()
        self.free_coords_pool = RandomSet(
            {Position(x, y) for x in range(size) for y in range(size)}
        )
        for segment in self.snake.occupied:
            self.free_coords_pool.discard(segment)
        self.food = self.free_coords_pool.pop_random(self.rng)

    def next_food(self) -> None:
        self.food = self.free_coords_pool.pop_random(self.rng)

    def change_direction(self, direction: Direction) -> bool:
        return self.snake.change_direction(direction)

    def game_state(self) -> GameState:
        direction = self.snake.direction
        head = self.snake.head
        tail = self.snake.segments[-1]
        occupied_ratio = len(self.snake) / (self.size**2)

        return GameState(
            direction=self._one_hot_direction(direction),
            blocked_left=float(self._is_blocked(direction.left())),
            blocked_forward=float(self._is_blocked(direction)),
            blocked_right=float(self._is_blocked(direction.right())),
            dx=self._shortest_delta(head.x, self.food.x),
            dy=self._shortest_delta(head.y, self.food.y),
            occupied_ratio=occupied_ratio,
            tail_dx=self._shortest_delta(head.x, tail.x),
            tail_dy=self._shortest_delta(head.y, tail.y),
        )

    def tick(self) -> bool:
        self.epoch += 1
        next_head = self.snake.next_head(self.snake.direction, self.size)
        will_eat = next_head == self.food
        if self.snake.would_collide(next_head, grow=will_eat):
            return False

        tail = self.snake.advance(next_head, grow=will_eat)

        if tail is not None:
            self.free_coords_pool.add(tail)
        self.free_coords_pool.discard(next_head)

        if will_eat:
            self.score += 1
            self.next_food()

        return True

    def state_img(self) -> np.ndarray:
        arr = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        arr[self.food.y, self.food.x] = [255, 0, 0]
        for pos in self.snake.segments:
            arr[pos.y, pos.x] = [255, 255, 255]
        return arr

    def _is_blocked(self, direction: Direction) -> bool:
        next_head = self.snake.next_head(direction, self.size)
        will_grow = next_head == self.food

        return self.snake.would_collide(next_head, grow=will_grow)

    def _one_hot_direction(
        self, direction: Direction
    ) -> tuple[float, float, float, float]:
        return (
            float(direction is Direction.UP),
            float(direction is Direction.RIGHT),
            float(direction is Direction.DOWN),
            float(direction is Direction.LEFT),
        )

    def _shortest_delta(self, start: int, target: int) -> float:
        delta = target - start
        half_size = self.size / 2

        if delta > half_size:
            delta -= self.size
        elif delta < -half_size:
            delta += self.size

        return delta / self.size

    def step(self, action: RelativeAction) -> tuple[GameState, float, bool]:
        if action is RelativeAction.LEFT:
            self.snake.change_direction(self.snake.direction.left())
        elif action is RelativeAction.RIGHT:
            self.snake.change_direction(self.snake.direction.right())

        score_before = self.score
        still_playing = self.tick()
        next_state = self.game_state()
        done = not still_playing

        if done:
            reward = -10.0
        elif self.score > score_before:
            reward = 3.0
        else:
            reward = -0.01

        return next_state, reward, done
