import torch
from torch import nn

from .noisy_linear import NoisyLinear


class DuelingDqn(nn.Module):
    def __init__(self, v_min: float, v_max: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value = nn.Sequential(
            NoisyLinear(64 * 11 * 11, 256), nn.ReLU(), NoisyLinear(256, 51)
        )

        self.advantage = nn.Sequential(
            NoisyLinear(64 * 11 * 11, 256), nn.ReLU(), NoisyLinear(256, 3 * 51)
        )

        self.register_buffer("support", torch.linspace(v_min, v_max, 51))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.network(x)
        value = self.value(base).reshape(-1, 1, 51)
        advantage = self.advantage(base).reshape(-1, 3, 51)
        logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        return torch.log_softmax(logits, dim=-1)

    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward(x).exp()
        return probs @ self.get_buffer("support")

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
