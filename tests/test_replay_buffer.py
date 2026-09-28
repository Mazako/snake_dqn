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

        states, actions, rewards, next_states, dones = buffer.sample(2)

        self.assertEqual(states.shape, (2, 3, 11, 11))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))
        self.assertEqual(next_states.shape, (2, 3, 11, 11))
        self.assertEqual(dones.shape, (2,))

    def test_uniform_sample_repeats_indices_and_seed_repeats_draws(self) -> None:
        first = ReplayBuffer(2, seed=42)
        second = ReplayBuffer(2, seed=42)
        state = torch.zeros((3, 11, 11))
        transitions = (
            Transition(state, RelativeAction.LEFT, 0.0, state, False),
            Transition(state, RelativeAction.RIGHT, 0.0, state, False),
        )
        for transition in transitions:
            first.append(transition)
            second.append(transition)

        counts = {RelativeAction.LEFT: 0, RelativeAction.RIGHT: 0}
        saw_repeated_index = False

        for _ in range(200):
            first_actions = first.sample(2)[1]
            second_actions = second.sample(2)[1]
            torch.testing.assert_close(first_actions, second_actions)
            saw_repeated_index |= bool(first_actions[0] == first_actions[1])
            for action in first_actions.tolist():
                counts[RelativeAction(action)] += 1

        self.assertTrue(saw_repeated_index)
        self.assertGreater(counts[RelativeAction.LEFT], 150)
        self.assertGreater(counts[RelativeAction.RIGHT], 150)
