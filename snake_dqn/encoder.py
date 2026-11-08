import torch
from torch import nn

ENCODED_CHANNELS = 16
HIDDEN = 256


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(128, ENCODED_CHANNELS, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[:-3]
        out = self.convs(x.reshape(-1, *x.shape[-3:]))
        return out.reshape(*batch, -1)


def make_trunk(size: int) -> nn.Sequential:
    return nn.Sequential(
        Encoder(),
        nn.Linear(ENCODED_CHANNELS * size * size, HIDDEN),
        nn.ReLU(),
    )
