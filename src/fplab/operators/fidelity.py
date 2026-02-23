from __future__ import annotations

from dataclasses import dataclass, field

import torch


class Fidelity:
    """Abstract data-fidelity interface f(x; y)."""

    def value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def lipschitz(self) -> float:
        raise NotImplementedError


@dataclass
class LeastSquaresFidelity(Fidelity):
    """Least-squares fidelity: f(x;y)=0.5*||Ax-y||^2."""

    A: torch.Tensor
    _lipschitz_cache: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.A.dim() != 2:
            raise ValueError("A must be a 2D matrix")

    def value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residual = self._apply_A(x) - y
        return 0.5 * torch.sum(residual * residual, dim=-1)

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        residual = self._apply_A(x) - y
        return residual @ self.A

    def lipschitz(self) -> float:
        # For f(x)=0.5||Ax-y||^2, grad-Lipschitz constant is ||A||_2^2.
        if self._lipschitz_cache is None:
            spectral = torch.linalg.matrix_norm(self.A, ord=2)
            self._lipschitz_cache = float((spectral * spectral).item())
        return self._lipschitz_cache

    def _apply_A(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T
