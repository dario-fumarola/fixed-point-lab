from __future__ import annotations

import math

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.proxgrad import ProxGradSolver


def test_pg_linesearch_failure_path_returns_finite_candidate() -> None:
    torch.manual_seed(0)

    dim = 1
    fidelity = LeastSquaresFidelity(A=torch.tensor([[25.0]]))
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(8, 8), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=50, lr=5e-2, tol=1e-6))
    solver = ProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    x = torch.tensor([[2.0]])
    y = torch.tensor([[1.0]])
    x_next, grad_norm, threshold, alpha, backtracks, accepted = solver.step(
        x=x,
        y=y,
        lam=0.1,
        differentiable=False,
        alpha_scale=1e6,
        line_search=True,
        max_backtracks=0,
    )

    assert not accepted
    assert torch.isfinite(x_next).all()
    assert math.isfinite(grad_norm)
    assert math.isfinite(threshold)
    assert alpha > 0.0
    assert backtracks == 0
