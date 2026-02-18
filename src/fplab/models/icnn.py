from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonNegativeLinear(nn.Module):
    """Linear layer with nonnegative weights by construction via softplus."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight_unconstrained = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features))

    @property
    def weight(self) -> torch.Tensor:
        return F.softplus(self.weight_unconstrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


@dataclass
class ICNNConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (64, 64)
    mu_quadratic: float = 1e-3


class ICNNRegularizer(nn.Module):
    """Input-convex neural regularizer R_theta(x)."""

    def __init__(self, config: ICNNConfig) -> None:
        super().__init__()
        if config.mu_quadratic <= 0:
            raise ValueError("mu_quadratic must be > 0 to keep prox subproblem strongly convex")

        self.config = config
        self.mu_quadratic = float(config.mu_quadratic)

        dims = (config.input_dim, *config.hidden_dims)
        self.x_layers = nn.ModuleList()
        self.z_layers = nn.ModuleList()

        for idx in range(len(config.hidden_dims)):
            self.x_layers.append(nn.Linear(dims[0], dims[idx + 1], bias=True))
            if idx == 0:
                self.z_layers.append(nn.Identity())
            else:
                self.z_layers.append(NonNegativeLinear(dims[idx], dims[idx + 1]))

        self.out_x = nn.Linear(dims[0], 1, bias=True)
        self.out_z = NonNegativeLinear(dims[-1], 1)

    def phi(self, x: torch.Tensor) -> torch.Tensor:
        """Convex ICNN core potential (without quadratic term)."""
        z = F.softplus(self.x_layers[0](x))
        for x_layer, z_layer in zip(self.x_layers[1:], self.z_layers[1:], strict=True):
            z = F.softplus(x_layer(x) + z_layer(z))
        out = self.out_x(x) + self.out_z(z)
        return out.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quad = 0.5 * self.mu_quadratic * torch.sum(x * x, dim=-1)
        return self.phi(x) + quad

    def convexity_sanity_check(self) -> bool:
        """Quick structural check: all z-weights are nonnegative by parameterization."""
        for layer in self.modules():
            if isinstance(layer, NonNegativeLinear):
                if torch.any(layer.weight < 0):
                    return False
        return True
