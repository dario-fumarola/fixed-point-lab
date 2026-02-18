from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class FixedPointTrace:
    residuals: list[float]


def krasnoselskii_mann(
    operator: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    relax: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, FixedPointTrace]:
    """Run Krasnoselskii-Mann iteration x_{k+1}=(1-lambda)x_k+lambda T(x_k)."""
    if relax <= 0.0 or relax > 1.0:
        raise ValueError("relax must be in (0, 1]")

    x = x0.clone()
    residuals: list[float] = []

    for _ in range(max_iter):
        tx = operator(x)
        x_next = (1.0 - relax) * x + relax * tx
        residual = float(torch.linalg.norm(x_next - x).item())
        residuals.append(residual)
        x = x_next

        if residual <= tol:
            break

    return x, FixedPointTrace(residuals=residuals)
