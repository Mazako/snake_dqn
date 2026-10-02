from __future__ import annotations

from random import Random

import torch

from snake_dqn.agent import RelativeAction

from .direction import Direction
from .end_reason import EndReason
from .position import Position
from .random_set import RandomSet
from .snake import Snake
from .state import GameState


class Game:
    epoch: int
    steps_since_apple: int
    end_reason: EndReason | None
    score: int
    snake: Snake
    food: Position

    def __init__(
        self,
        size: int,
        seed: int | None = None,
        hunger_base: int | None = None,
        hunger_per_segment: int = 2,
    ) -> None:
        if size < 3 or size % 2 == 0:
            raise ValueError("size must be an odd integer at least 3")
        if hunger_base is not None and hunger_base <= 0:
            raise ValueError("hunger_base must be positive")
        if hunger_per_segment < 0:
            raise ValueError("hunger_per_segment must not be negative")

        self.size = size
        self.hunger_base = 2 * size if hunger_base is None else hunger_base
        self.hunger_per_segment = hunger_per_segment
        self.score = 0
        self.epoch = 0
        self.steps_since_apple = 0
        self.end_reason = None
        self.snake = Snake(Position(0, 0), Direction.RIGHT)
        self.rng = Random(seed)
        self.free_coords_pool = RandomSet(
            Position(x, y) for y in range(size) for x in range(size)
        )
        for segment in self.snake.occupied:
            self.free_coords_pool.discard(segment)
        self.food = self.free_coords_pool.pop_random(self.rng)

    @property
    def hunger_limit(self) -> int:
        return self.hunger_base + self.hunger_per_segment * len(self.snake)

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
            self.steps_since_apple = 0
            if not self.free_coords_pool:
                return False
            self.next_food()
        else:
            self.steps_since_apple += 1

        return True

    def state_img(self) -> torch.Tensor:
        img = torch.zeros((3, self.size, self.size), dtype=torch.float32)
        for pos in self.snake.segments:
            img[0, pos.y, pos.x] = 1.0
        img[1, self.food.y, self.food.x] = 1.0
        tail = self.snake.segments[-1]
        img[2, tail.y, tail.x] = 1.0

        head = self.snake.head
        center_x = self.size // 2
        center_y = self.size // 2
        img = torch.roll(
            img,
            shifts=(center_y - head.y, center_x - head.x),
            dims=(-2, -1),
        )

        match self.snake.direction:
            case Direction.UP:
                rotations = 0
            case Direction.RIGHT:
                rotations = 1
            case Direction.DOWN:
                rotations = 2
            case Direction.LEFT:
                rotations = 3

        return torch.rot90(img, k=rotations, dims=(-2, -1))

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
        starved = self.steps_since_apple >= self.hunger_limit
        won = not still_playing and not self.free_coords_pool

        if won:
            self.end_reason = EndReason.WIN
        elif not still_playing:
            self.end_reason = EndReason.COLLISION
        elif starved:
            self.end_reason = EndReason.STARVATION
        done = self.end_reason is not None

        if won:
            reward = 3.0
        elif done:
            reward = -10.0
        elif self.score > score_before:
            reward = 3.0
        else:
            reward = -0.01

        return next_state, reward, done
