from __future__ import annotations

import torch

from fplab.solvers.fixed_point import krasnoselskii_mann


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
