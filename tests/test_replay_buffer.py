import unittest

import torch

from snake_dqn.agent import RelativeAction
from snake_dqn.replay_buffer import ReplayBuffer, Transition


class ReplayBufferTests(unittest.TestCase):
    def test_sample_returns_image_batches(self) -> None:
        buffer = ReplayBuffer(2)
        state = torch.zeros((3, 11, 11))
        next_state = torch.ones((3, 11, 11))

        buffer.append(Transition(state, RelativeAction.LEFT, -0.01, next_state, False))
        buffer.append(Transition(next_state, RelativeAction.RIGHT, 3.0, state, True))

        states, actions, rewards, next_states, dones, weights, indices = buffer.sample(
            2
        )

        self.assertEqual(states.shape, (2, 3, 11, 11))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(next_states.shape, (2, 3, 11, 11))
        self.assertEqual(dones.shape, (2,))
        self.assertEqual(weights.shape, (2,))
        self.assertEqual(indices.shape, (2,))

    def test_sample_uses_priorities_and_returns_weights_for_drawn_indices(self) -> None:
        buffer = ReplayBuffer(4, beta=1.0)
        state = torch.zeros((3, 11, 11))
        transition = Transition(state, RelativeAction.LEFT, 0.0, state, False)
        buffer.append(transition)
        buffer.append(transition)
        buffer.update_priorities(torch.tensor([1.0, 3.0]), torch.tensor([0, 1]))

        priorities = torch.tensor(
            [1.0 + buffer.priority_eps, 3.0 + buffer.priority_eps]
        )
        probabilities = priorities / priorities.sum()
        saw_repeated_index = False

        for _ in range(20):
            *_, weights, indices = buffer.sample(2)
            expected_weights = (len(buffer) * probabilities[indices]).reciprocal()
            torch.testing.assert_close(weights, expected_weights)
            saw_repeated_index |= bool(indices[0] == indices[1])

        self.assertTrue(saw_repeated_index)
