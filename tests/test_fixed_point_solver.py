from __future__ import annotations

import torch

from fplab.solvers.fixed_point import anderson_acceleration, krasnoselskii_mann


def test_krasnoselskii_mann_converges_on_contraction() -> None:
    torch.manual_seed(0)
    b = torch.randn(4)

    def operator(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x + b

    x0 = torch.zeros_like(b)
    x_star = 2.0 * b

    x_hat, trace = krasnoselskii_mann(operator=operator, x0=x0, relax=1.0, max_iter=80, tol=1e-8)

    assert len(trace.residuals) > 1
    assert torch.allclose(x_hat, x_star, atol=1e-5)
    assert trace.residuals[-1] <= trace.residuals[0]


def test_anderson_acceleration_converges_on_contraction() -> None:
    torch.manual_seed(0)
    b = torch.randn(4)

    def operator(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x + b

    x0 = torch.zeros_like(b)
    x_star = 2.0 * b

    x_hat, trace = anderson_acceleration(operator=operator, x0=x0, history=4, max_iter=30, tol=1e-8)

    assert len(trace.residuals) > 1
    assert torch.allclose(x_hat, x_star, atol=1e-5)
    assert trace.residuals[-1] <= trace.residuals[0]


def test_anderson_is_not_worse_than_km_on_linear_contraction() -> None:
    torch.manual_seed(0)
    n = 5
    b = torch.randn(n)
    B = 0.6 * torch.eye(n)

    def operator(x: torch.Tensor) -> torch.Tensor:
        return x @ B.T + b

    x0 = torch.zeros_like(b)

    _x_km, trace_km = krasnoselskii_mann(operator=operator, x0=x0, relax=1.0, max_iter=25, tol=1e-10)
    _x_and, trace_and = anderson_acceleration(operator=operator, x0=x0, history=5, max_iter=25, tol=1e-10)

    assert trace_and.residuals[-1] <= trace_km.residuals[-1] + 1e-8
