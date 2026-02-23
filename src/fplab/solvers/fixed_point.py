from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class FixedPointTrace:
    residuals: list[float]


def _solve_anderson_coefficients(F: torch.Tensor, reg: float) -> torch.Tensor:
    m = F.shape[1]
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
        return coeff
    except RuntimeError:
        coeff = torch.zeros((m,), device=F.device, dtype=F.dtype)
        coeff[-1] = 1.0
        return coeff


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
            if g.ndim == 1:
                F = torch.stack([fi.reshape(-1) for fi in f_hist], dim=1)  # (n, m)
                coeff = _solve_anderson_coefficients(F, reg=reg)
                g_stack = torch.stack(g_hist, dim=0)  # (m, n)
                x_anderson = torch.tensordot(coeff, g_stack, dims=([0], [0]))
            else:
                # Batched inputs: solve one small KKT system per sample (batched linear algebra)
                # to avoid cross-sample coupling while keeping Python overhead low.
                m = len(g_hist)
                g_shape = g.shape
                batch_size = g_shape[0]

                f_stack = torch.stack([fi.reshape(batch_size, -1) for fi in f_hist], dim=0)  # (m, b, n)
                g_stack = torch.stack([gi.reshape(batch_size, -1) for gi in g_hist], dim=0)  # (m, b, n)
                f_batch = f_stack.permute(1, 2, 0)  # (b, n, m)

                eye = torch.eye(m, device=f_batch.device, dtype=f_batch.dtype).expand(batch_size, m, m)
                H = torch.matmul(f_batch.transpose(1, 2), f_batch) + reg * eye  # (b, m, m)
                ones = torch.ones((batch_size, m, 1), device=f_batch.device, dtype=f_batch.dtype)

                kkt = torch.zeros((batch_size, m + 1, m + 1), device=f_batch.device, dtype=f_batch.dtype)
                kkt[:, :m, :m] = H
                kkt[:, :m, m:] = ones
                kkt[:, m:, :m] = ones.transpose(1, 2)

                rhs = torch.zeros((batch_size, m + 1, 1), device=f_batch.device, dtype=f_batch.dtype)
                rhs[:, m, 0] = 1.0

                try:
                    sol = torch.linalg.solve(kkt, rhs)
                    coeffs = sol[:, :m, 0]  # (b, m)
                    finite_mask = torch.isfinite(coeffs).all(dim=1)
                except RuntimeError:
                    coeffs = torch.empty((batch_size, m), device=f_batch.device, dtype=f_batch.dtype)
                    finite_mask = torch.zeros((batch_size,), device=f_batch.device, dtype=torch.bool)

                if not bool(torch.all(finite_mask)):
                    for batch_idx in range(batch_size):
                        if bool(finite_mask[batch_idx]):
                            continue
                        F_batch = f_batch[batch_idx]
                        coeffs[batch_idx] = _solve_anderson_coefficients(F_batch, reg=reg)

                g_batch = g_stack.permute(1, 0, 2)  # (b, m, n)
                x_anderson_flat = torch.sum(coeffs.unsqueeze(-1) * g_batch, dim=1)  # (b, n)
                x_anderson = x_anderson_flat.reshape(g_shape)

            x_next = (1.0 - damping) * x + damping * x_anderson

        residual = float(torch.linalg.norm(x_next - x).item())
        residuals.append(residual)
        x = x_next

        if residual <= tol:
            break

    return x, FixedPointTrace(residuals=residuals)
