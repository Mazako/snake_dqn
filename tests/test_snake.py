import unittest
from collections import deque

from snake_dqn.direction import Direction
from snake_dqn.position import Position
from snake_dqn.snake import Snake


class SnakeTests(unittest.TestCase):
    def test_would_collide_ignores_tail_without_growth(self) -> None:
        snake = Snake(Position(5, 5), Direction.RIGHT)
        snake.segments = deque([Position(5, 5), Position(6, 5), Position(7, 5)])
        snake.occupied = set(snake.segments)

        tail = snake.segments[-1]

        self.assertFalse(snake.would_collide(tail, grow=False))
        self.assertTrue(snake.would_collide(tail, grow=True))

    def test_advance_returns_released_tail(self) -> None:
        snake = Snake(Position(5, 5), Direction.RIGHT)
        snake.segments = deque([Position(5, 5), Position(4, 5), Position(3, 5)])
        snake.occupied = set(snake.segments)

        released_tail = snake.advance(Position(6, 5), grow=False)

        self.assertEqual(released_tail, Position(3, 5))
        self.assertEqual(
            list(snake.segments), [Position(6, 5), Position(5, 5), Position(4, 5)]
        )
        self.assertEqual(snake.occupied, set(snake.segments))
