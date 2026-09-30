from math import sqrt

import torch
import torch.nn.functional as F
from torch import nn


class NoisyLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, std_init: float = 0.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        r = sqrt(1 / self.in_features)
        sig = self.std_init / sqrt(self.in_features)
        with torch.no_grad():
            self.weight_mu.uniform_(-r, r)
            self.bias_mu.uniform_(-r, r)
            self.weight_sigma.fill_(sig)
            self.bias_sigma.fill_(sig)

    def reset_noise(self):
        dev = self.get_buffer("weight_epsilon").device
        eps_in = self._scale_noise(self.in_features, dev)
        eps_out = self._scale_noise(self.out_features, dev)
        self.get_buffer("weight_epsilon").copy_(torch.outer(eps_out, eps_in))
        self.get_buffer("bias_epsilon").copy_(eps_out)

    def _scale_noise(self, size: int, device: torch.device) -> torch.Tensor:
        f = torch.randn(size, device=device)
        return f.sign() * f.abs().sqrt()

    def forward(self, x):
        if not self.training:
            w = self.weight_mu
            b = self.bias_mu
        else:
            w = self.weight_mu + self.weight_sigma * self.get_buffer("weight_epsilon")
            b = self.bias_mu + self.bias_sigma * self.get_buffer("bias_epsilon")
        return F.linear(x, w, b)
