import unittest
from collections import deque

from snake_dqn.direction import Direction
from snake_dqn.game import Game
from snake_dqn.position import Position


class GameStateTests(unittest.TestCase):
    def test_game_rejects_even_board_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "odd integer"):
            Game(10)

    def test_state_contains_one_hot_direction_and_food_delta(self) -> None:
        game = Game(11)
        game.snake.direction = Direction.RIGHT
        game.snake.segments = deque([Position(5, 5)])
        game.snake.occupied = set(game.snake.segments)
        game.food = Position(7, 3)

        state = game.game_state()

        self.assertEqual(state.direction, (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(state.blocked_left, 0.0)
        self.assertEqual(state.blocked_forward, 0.0)
        self.assertEqual(state.blocked_right, 0.0)
        self.assertEqual(state.dx, 2 / 11)
        self.assertEqual(state.dy, -2 / 11)
        self.assertEqual(state.occupied_ratio, 1 / 121)
        self.assertEqual(state.tail_dx, 0.0)
        self.assertEqual(state.tail_dy, 0.0)

    def test_state_marks_obstacle_in_front(self) -> None:
        game = Game(11)
        game.snake.direction = Direction.RIGHT
        game.snake.segments = deque([Position(5, 5), Position(6, 5), Position(7, 5)])
        game.snake.occupied = set(game.snake.segments)
        game.food = Position(0, 0)

        state = game.game_state()

        self.assertEqual(state.blocked_left, 0.0)
        self.assertEqual(state.blocked_forward, 1.0)
        self.assertEqual(state.blocked_right, 0.0)
        self.assertEqual(state.occupied_ratio, 3 / 121)
        self.assertEqual(state.tail_dx, 2 / 11)
        self.assertEqual(state.tail_dy, 0.0)

    def test_state_uses_shortest_toroidal_food_delta(self) -> None:
        game = Game(11)
        game.snake.segments = deque([Position(0, 0)])
        game.snake.occupied = set(game.snake.segments)
        game.food = Position(10, 1)

        state = game.game_state()

        self.assertEqual(state.dx, -1 / 11)
        self.assertEqual(state.dy, 1 / 11)
