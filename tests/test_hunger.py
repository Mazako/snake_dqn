import unittest
from collections import deque

from snake_dqn.agent import RelativeAction
from snake_dqn.end_reason import EndReason
from snake_dqn.game import Game
from snake_dqn.position import Position


class HungerTests(unittest.TestCase):
    def test_default_limit_scales_with_board_and_snake_length(self) -> None:
        game = Game(11)

        self.assertEqual(game.hunger_limit, 2 * 11 + 2 * 1)

    def test_limit_grows_with_snake_length(self) -> None:
        game = Game(11, hunger_base=10, hunger_per_segment=3)
        self.assertEqual(game.hunger_limit, 13)

        game.snake.segments = deque([Position(0, 0), Position(0, 1), Position(0, 2)])

        self.assertEqual(game.hunger_limit, 19)

    def test_hunger_grows_by_one_per_step_without_apple(self) -> None:
        game = Game(11)
        game.food = Position(5, 5)

        game.step(RelativeAction.FORWARD)
        game.step(RelativeAction.FORWARD)

        self.assertEqual(game.steps_since_apple, 2)

    def test_snake_starves_exactly_at_limit(self) -> None:
        game = Game(11, hunger_base=3, hunger_per_segment=0)
        game.food = Position(5, 5)

        results = [game.step(RelativeAction.FORWARD) for _ in range(3)]

        self.assertEqual([done for _, _, done in results], [False, False, True])
        self.assertEqual(results[0][1], -0.01)
        self.assertEqual(results[-1][1], -10.0)
        self.assertIs(game.end_reason, EndReason.STARVATION)

    def test_end_reason_is_none_while_game_is_running(self) -> None:
        game = Game(11)
        game.food = Position(5, 5)

        game.step(RelativeAction.FORWARD)

        self.assertIsNone(game.end_reason)

    def test_hitting_own_body_ends_with_collision(self) -> None:
        game = Game(11)
        game.snake.segments = deque([Position(5, 5), Position(6, 5), Position(7, 5)])
        game.snake.occupied = set(game.snake.segments)
        game.food = Position(0, 0)

        _, reward, done = game.step(RelativeAction.FORWARD)

        self.assertTrue(done)
        self.assertEqual(reward, -10.0)
        self.assertIs(game.end_reason, EndReason.COLLISION)

    def test_eating_on_last_allowed_step_saves_snake(self) -> None:
        game = Game(11, hunger_base=3, hunger_per_segment=0)
        game.food = Position(3, 0)

        results = [game.step(RelativeAction.FORWARD) for _ in range(3)]

        self.assertFalse(results[-1][2])
        self.assertIsNone(game.end_reason)
        self.assertEqual(results[-1][1], 3.0)
        self.assertEqual(game.steps_since_apple, 0)

    def test_eating_resets_hunger_counter(self) -> None:
        game = Game(11, hunger_base=3, hunger_per_segment=0)
        game.food = Position(2, 0)
        game.step(RelativeAction.FORWARD)
        game.step(RelativeAction.FORWARD)
        game.food = Position(5, 5)

        results = [game.step(RelativeAction.FORWARD) for _ in range(3)]

        self.assertEqual([done for _, _, done in results], [False, False, True])

    def test_rejects_invalid_hunger_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "hunger_base"):
            Game(11, hunger_base=0)
        with self.assertRaisesRegex(ValueError, "hunger_per_segment"):
            Game(11, hunger_per_segment=-1)


if __name__ == "__main__":
    unittest.main()
