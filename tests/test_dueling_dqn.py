import unittest

import torch

from snake_dqn.dueling_dqn import DuelingDqn

V_MIN = -10.0
V_MAX = 30.0
N_ACTIONS = 3
N_ATOMS = 51
BATCH_SIZES = (1, 2, 64)


class DuelingDqnTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = DuelingDqn(V_MIN, V_MAX)

    def states(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, 3, 11, 11)

    def test_forward_returns_distribution_per_action(self) -> None:
        for batch_size in BATCH_SIZES:
            with self.subTest(batch_size=batch_size):
                log_probs = self.model(self.states(batch_size))

                self.assertEqual(log_probs.shape, (batch_size, N_ACTIONS, N_ATOMS))

    def test_forward_returns_log_probabilities(self) -> None:
        log_probs = self.model(self.states(8))

        self.assertTrue((log_probs <= 0).all())
        torch.testing.assert_close(
            log_probs.exp().sum(dim=-1), torch.ones(8, N_ACTIONS)
        )

    def test_samples_in_batch_do_not_affect_each_other(self) -> None:
        self.model.eval()
        states = self.states(4)

        batched = self.model(states)

        for i, state in enumerate(states):
            torch.testing.assert_close(batched[i], self.model(state.unsqueeze(0))[0])

    def test_support_spans_v_min_to_v_max_evenly(self) -> None:
        torch.testing.assert_close(
            self.model.support, torch.linspace(V_MIN, V_MAX, N_ATOMS)
        )

    def test_support_is_buffer_not_parameter(self) -> None:
        self.assertIn("support", self.model.state_dict())
        self.assertNotIn("support", (name for name, _ in self.model.named_parameters()))

    def test_q_values_have_one_value_per_action(self) -> None:
        for batch_size in BATCH_SIZES:
            with self.subTest(batch_size=batch_size):
                q_values = self.model.q_values(self.states(batch_size))

                self.assertEqual(q_values.shape, (batch_size, N_ACTIONS))

    def test_q_values_are_expected_return_of_distribution(self) -> None:
        states = self.states(8)

        expected = self.model(states).exp() @ self.model.support

        torch.testing.assert_close(self.model.q_values(states), expected)

    def test_q_values_stay_within_support(self) -> None:
        q_values = self.model.q_values(self.states(64))

        self.assertTrue((q_values >= V_MIN).all())
        self.assertTrue((q_values <= V_MAX).all())

    def test_backward_gives_gradients_to_every_parameter(self) -> None:
        self.model(self.states(8)).sum().backward()

        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
