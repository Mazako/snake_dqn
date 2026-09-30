from math import sqrt

import torch
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

    def reset_parameters(self):
        r = sqrt(1 / self.in_features)
        sig = self.std_init / sqrt(self.in_features)
        with torch.no_grad():
            self.weight_mu.uniform_(-r, r)
            self.bias_mu.uniform_(-r, r)
            self.weight_sigma.fill_(sig)
            self.bias_sigma.fill_(sig)
