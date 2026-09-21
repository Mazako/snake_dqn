import unittest
from random import Random

from snake_dqn.agent import RandomAgent, RelativeAction
from snake_dqn.dqn_state import GameState


class RandomAgentTests(unittest.TestCase):
    def test_choose_action_returns_relative_action(self) -> None:
        agent = RandomAgent(Random(42))
        state = GameState(
            direction=(0.0, 1.0, 0.0, 0.0),
            blocked_left=0.0,
            blocked_forward=0.0,
            blocked_right=0.0,
            dx=0.2,
            dy=-0.1,
            occupied_ratio=0.2,
            tail_dy=1.0,
            tail_dx=0.2,
        )

        action = agent.choose_action(state)

        self.assertIsInstance(action, RelativeAction)
