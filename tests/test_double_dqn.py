import unittest

import torch

from snake_dqn.double_dqn import (
    estimate_support_range,
    project_distribution,
    train_step,
)
from snake_dqn.dueling_dqn import DuelingDqn
from snake_dqn.replay_buffer import ReplayBuffer, Transition

SUPPORT = torch.linspace(-10.0, 10.0, 21)
N_ATOMS = SUPPORT.numel()


def one_hot(index: int) -> torch.Tensor:
    probs = torch.zeros(1, N_ATOMS)
    probs[0, index] = 1.0
    return probs


def project(
    next_probs: torch.Tensor, reward: float, done: bool, discount: float = 0.9
) -> torch.Tensor:
    return project_distribution(
        next_probs,
        torch.tensor([reward]),
        torch.tensor([float(done)]),
        SUPPORT,
        discount,
    )


class ProjectDistributionTests(unittest.TestCase):
    def test_projected_distribution_sums_to_one(self) -> None:
        torch.manual_seed(0)
        batch_size = 32
        next_probs = torch.softmax(torch.randn(batch_size, N_ATOMS), dim=1)
        rewards = torch.randn(batch_size) * 20
        dones = (torch.rand(batch_size) < 0.3).float()

        projected = project_distribution(next_probs, rewards, dones, SUPPORT, 0.9)

        self.assertTrue((projected >= 0).all())
        torch.testing.assert_close(projected.sum(dim=1), torch.ones(batch_size))

    def test_terminal_reward_on_atom_puts_all_mass_there(self) -> None:
        projected = project(one_hot(0), reward=3.0, done=True)

        torch.testing.assert_close(projected, one_hot(13))

    def test_terminal_reward_between_atoms_splits_mass(self) -> None:
        projected = project(one_hot(0), reward=2.25, done=True)

        self.assertAlmostEqual(projected[0, 12].item(), 0.75)
        self.assertAlmostEqual(projected[0, 13].item(), 0.25)

    def test_shift_landing_exactly_on_atom_keeps_mass(self) -> None:
        projected = project(one_hot(15), reward=0.0, done=False, discount=1.0)

        torch.testing.assert_close(projected, one_hot(15))

    def test_returns_beyond_support_are_clipped_to_edges(self) -> None:
        self.assertEqual(project(one_hot(0), 100.0, done=True)[0, -1].item(), 1.0)
        self.assertEqual(project(one_hot(0), -100.0, done=True)[0, 0].item(), 1.0)

    def test_expected_value_follows_bellman_inside_support(self) -> None:
        next_probs = torch.softmax(torch.randn(1, N_ATOMS) * 0.1, dim=1)
        reward, discount = 0.7, 0.5

        projected = project(next_probs, reward, done=False, discount=discount)

        torch.testing.assert_close(
            projected @ SUPPORT, reward + discount * (next_probs @ SUPPORT)
        )


class TrainStepTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = DuelingDqn(-10.0, 30.0)
        self.target_model = DuelingDqn(-10.0, 30.0)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(64)
        for i in range(16):
            self.buffer.append(
                Transition(
                    torch.rand(3, 11, 11),
                    i % 3,
                    -10.0 if i % 4 == 0 else 3.0,
                    torch.rand(3, 11, 11),
                    i % 4 == 0,
                )
            )

    def step(self, batch_size: int) -> float | None:
        return train_step(
            self.model,
            self.target_model,
            self.buffer,
            batch_size,
            0.99,
            self.optimizer,
            torch.device("cpu"),
        )

    def test_returns_none_until_buffer_has_batch(self) -> None:
        self.assertIsNone(self.step(batch_size=32))

    def test_returns_positive_cross_entropy(self) -> None:
        loss = self.step(batch_size=8)

        assert loss is not None
        self.assertGreater(loss, 0.0)

    def test_updates_online_model_but_not_target(self) -> None:
        online_before = [p.detach().clone() for p in self.model.parameters()]
        target_before = [p.detach().clone() for p in self.target_model.parameters()]

        self.step(batch_size=8)

        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(online_before, self.model.parameters())
            )
        )
        for old, new in zip(target_before, self.target_model.parameters()):
            torch.testing.assert_close(old, new)


class EstimateSupportRangeTests(unittest.TestCase):
    def test_v_min_is_immediate_death(self) -> None:
        v_min, _ = estimate_support_range(0.99, 3.0, -0.01, -10.0, 10)

        self.assertEqual(v_min, -10.0)


if __name__ == "__main__":
    unittest.main()
