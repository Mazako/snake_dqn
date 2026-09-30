import unittest

import torch

from snake_dqn.noisy_linear import NoisyLinear

IN_FEATURES = 16
OUT_FEATURES = 4
BATCH_SIZE = 8


class NoisyLinearTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.layer = NoisyLinear(IN_FEATURES, OUT_FEATURES)
        self.x = torch.randn(BATCH_SIZE, IN_FEATURES)

    def test_forward_returns_batch_of_out_features(self) -> None:
        self.assertEqual(self.layer(self.x).shape, (BATCH_SIZE, OUT_FEATURES))

    def test_has_four_learnable_parameters_mu_and_sigma_for_weight_and_bias(
        self,
    ) -> None:
        parameters = list(self.layer.parameters())

        self.assertEqual(len(parameters), 4)
        self.assertTrue(all(p.requires_grad for p in parameters))
        self.assertEqual(
            sorted(p.numel() for p in parameters),
            sorted(
                [
                    OUT_FEATURES * IN_FEATURES,
                    OUT_FEATURES * IN_FEATURES,
                    OUT_FEATURES,
                    OUT_FEATURES,
                ]
            ),
        )

    def test_noise_is_stored_in_buffers_not_parameters(self) -> None:
        buffers = list(self.layer.buffers())

        self.assertGreater(len(buffers), 0)
        self.assertTrue(all(not b.requires_grad for b in buffers))

    def test_forward_is_deterministic_until_noise_is_reset(self) -> None:
        self.layer.train()

        torch.testing.assert_close(self.layer(self.x), self.layer(self.x))

    def test_reset_noise_changes_output_in_train_mode(self) -> None:
        self.layer.train()
        before = self.layer(self.x)

        self.layer.reset_noise()

        self.assertFalse(torch.allclose(before, self.layer(self.x)))

    def test_reset_noise_does_not_change_parameters(self) -> None:
        before = [p.detach().clone() for p in self.layer.parameters()]

        self.layer.reset_noise()

        for old, new in zip(before, self.layer.parameters(), strict=True):
            torch.testing.assert_close(old, new)

    def test_eval_mode_ignores_noise(self) -> None:
        self.layer.eval()
        before = self.layer(self.x)

        self.layer.reset_noise()

        torch.testing.assert_close(before, self.layer(self.x))

    def test_same_noise_is_shared_across_batch(self) -> None:
        self.layer.train()
        batch = self.x[:1].repeat(BATCH_SIZE, 1)

        out = self.layer(batch)

        torch.testing.assert_close(out, out[:1].expand_as(out))

    def test_noisy_output_averages_to_eval_output(self) -> None:
        samples = 2000
        x = self.x[:1]
        self.layer.eval()
        mean_output = self.layer(x)

        self.layer.train()
        outputs = []
        for _ in range(samples):
            self.layer.reset_noise()
            outputs.append(self.layer(x))
        noisy_mean = torch.stack(outputs).mean(dim=0)

        torch.testing.assert_close(noisy_mean, mean_output, atol=0.1, rtol=0.0)

    def test_noise_has_nonzero_variance(self) -> None:
        outputs = []
        for _ in range(100):
            self.layer.reset_noise()
            outputs.append(self.layer(self.x[:1]))

        self.assertGreater(torch.stack(outputs).std(dim=0).min().item(), 0.0)

    def test_backward_gives_gradients_to_every_parameter_and_none_to_buffers(
        self,
    ) -> None:
        self.layer.train()

        self.layer(self.x).sum().backward()

        for parameter in self.layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0) # type: ignore
        for buffer in self.layer.buffers():
            self.assertIsNone(buffer.grad)

    def test_state_dict_contains_noise_and_restores_train_output(self) -> None:
        self.layer.train()
        other = NoisyLinear(IN_FEATURES, OUT_FEATURES)
        other.train()

        other.load_state_dict(self.layer.state_dict())

        torch.testing.assert_close(other(self.x), self.layer(self.x))

    def test_optimizer_step_updates_parameters(self) -> None:
        optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
        before = [p.detach().clone() for p in self.layer.parameters()]

        self.layer(self.x).pow(2).sum().backward()
        optimizer.step()

        for old, new in zip(before, self.layer.parameters(), strict=True):
            self.assertFalse(torch.allclose(old, new))


if __name__ == "__main__":
    unittest.main()
