from typing import Any

import torch
from torch import nn
from torchrl.modules import NoisyLinear


def make_q_network(size: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode="circular"),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode="circular"),
        nn.ReLU(),
        nn.Flatten(start_dim=-3),
        NoisyLinear(128 * size * size, 256),
        nn.ReLU(),
        NoisyLinear(256, 3),
    )


class RainbowQNetwork(nn.Module):
    def __init__(self, size: int, n_atoms: int = 51) -> None:
        super().__init__()
        self.n_atoms = n_atoms
        self.size = size
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(start_dim=-3),
        )
        self.value = nn.Sequential(
            NoisyLinear(128 * size * size, 256), nn.ReLU(), NoisyLinear(256, 1)
        )
        self.advantage = nn.Sequential(
            NoisyLinear(128 * size * size, 256), nn.ReLU(), NoisyLinear(256, 3)
        )

    def forward(self, x: torch.Tensor):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=-2, keepdim=True)
