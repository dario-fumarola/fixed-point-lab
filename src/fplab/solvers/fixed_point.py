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


def anderson_acceleration(
    operator: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    history: int = 5,
    damping: float = 1.0,
    reg: float = 1e-4,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, FixedPointTrace]:
    """Anderson acceleration for fixed-point iteration x = T(x).

    Uses constrained least-squares coefficients on recent residuals with
    regularized normal equations.
    """
    if history < 1:
        raise ValueError("history must be >= 1")
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in (0, 1]")
    if reg < 0.0:
        raise ValueError("reg must be nonnegative")

    x = x0.clone()
    residuals: list[float] = []
    g_hist: list[torch.Tensor] = []
    f_hist: list[torch.Tensor] = []

    for _ in range(max_iter):
        g = operator(x)
        f = g - x

        g_hist.append(g)
        f_hist.append(f)
        if len(g_hist) > history:
            g_hist.pop(0)
            f_hist.pop(0)

        if len(g_hist) == 1:
            x_next = g
        else:
            m = len(g_hist)
            F = torch.stack([fi.reshape(-1) for fi in f_hist], dim=1)  # (n, m)
            H = F.T @ F + reg * torch.eye(m, device=F.device, dtype=F.dtype)
            ones = torch.ones((m, 1), device=F.device, dtype=F.dtype)

            # Solve:
            # [H 1][c] = [0]
            # [1^T 0][u]   [1]
            kkt = torch.zeros((m + 1, m + 1), device=F.device, dtype=F.dtype)
            kkt[:m, :m] = H
            kkt[:m, m:] = ones
            kkt[m:, :m] = ones.T

            rhs = torch.zeros((m + 1, 1), device=F.device, dtype=F.dtype)
            rhs[m, 0] = 1.0

            try:
                sol = torch.linalg.solve(kkt, rhs)
                coeff = sol[:m, 0]
                if not torch.isfinite(coeff).all():
                    raise RuntimeError("non-finite Anderson coefficients")
            except RuntimeError:
                coeff = torch.zeros((m,), device=F.device, dtype=F.dtype)
                coeff[-1] = 1.0

            g_stack = torch.stack(g_hist, dim=0)  # (m, *shape)
            x_anderson = torch.tensordot(coeff, g_stack, dims=([0], [0]))
            x_next = (1.0 - damping) * x + damping * x_anderson

        residual = float(torch.linalg.norm(x_next - x).item())
        residuals.append(residual)
        x = x_next

        if residual <= tol:
            break

    return x, FixedPointTrace(residuals=residuals)
