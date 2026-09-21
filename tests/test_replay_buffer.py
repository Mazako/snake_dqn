import unittest
from random import Random

import torch

from snake_dqn.agent import RelativeAction
from snake_dqn.replay_buffer import ReplayBuffer, Transition


class ReplayBufferTests(unittest.TestCase):
    def test_sample_returns_image_batches(self) -> None:
        buffer = ReplayBuffer(2, Random(42))
        state = torch.zeros((2, 11, 11))
        next_state = torch.ones((2, 11, 11))

        buffer.append(Transition(state, RelativeAction.LEFT, -0.01, next_state, False))
        buffer.append(Transition(next_state, RelativeAction.RIGHT, 3.0, state, True))

        states, actions, rewards, next_states, dones = buffer.sample(2)

        self.assertEqual(states.shape, (2, 2, 11, 11))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(next_states.shape, (2, 2, 11, 11))
        self.assertEqual(dones.shape, (2,))
