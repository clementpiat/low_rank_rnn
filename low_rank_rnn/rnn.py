import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankRNN(nn.Module):
    def __init__(self, rank: int = 2, neurons: int = 1_000, input_dim: int = 1) -> None:
        super().__init__()

        # hyperparameters
        self.R = rank
        self.N = neurons
        self.P = input_dim

        # parameters
        self.m = nn.Linear(self.R, self.N, bias=False)
        self.n = nn.Linear(self.N, self.R, bias=False)
        self.I = nn.Linear(self.P, self.N, bias=False)
        self.w = nn.Linear(self.N, 1, bias=False)

        # init weights
        std = 1 / math.sqrt(self.N)
        nn.init.normal_(self.m.weight, std=std)
        nn.init.normal_(self.n.weight, std=std)
        nn.init.normal_(self.I.weight, std=std)
        nn.init.normal_(self.w.weight, std=std)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        assert u.dim() == 3, f"input u should have 3 dimensions, got {u.dim()}"
        assert u.size(2) == self.P, (
            f"input u should be of shape (B, T, {self.P}), got {u.shape}"
        )

        o: list[torch.Tensor] = []
        x_t = torch.zeros(self.N)
        for u_t in u:
            x_t = self.m(self.n(F.tanh(x_t))) + self.I(u_t)
            o_t = self.w(x_t)
            o.append(o_t)

        return torch.stack(o)
