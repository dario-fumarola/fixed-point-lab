from __future__ import annotations

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.proxgrad import ProxGradSolver


def test_pg_linesearch_recovers_from_aggressive_initial_step_size() -> None:
    torch.manual_seed(0)

    dim = 8
    A = torch.eye(dim)
    fidelity = LeastSquaresFidelity(A=A)
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(16, 16), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=120, lr=5e-2, tol=1e-6))
    solver = ProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    x0 = torch.randn(6, dim)
    y = torch.randn(6, dim)
    _x_hat, trace = solver.solve(
        x0=x0,
        y=y,
        lam=0.1,
        max_iter=15,
        tol=1e-6,
        alpha_scale=8.0,
        line_search=True,
    )

    assert len(trace.objectives) >= 2
    for prev, curr in zip(trace.objectives, trace.objectives[1:]):
        assert curr <= prev + 2e-4
    assert any(bt > 0 for bt in trace.backtracks)
    assert all(alpha > 0 for alpha in trace.step_sizes)
    assert len(trace.line_search_accepted) == len(trace.objectives)
    assert all(isinstance(item, bool) for item in trace.line_search_accepted)


def test_pg_linesearch_disallowed_in_differentiable_mode() -> None:
    torch.manual_seed(0)

    dim = 4
    A = torch.eye(dim)
    fidelity = LeastSquaresFidelity(A=A)
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(8, 8), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=20, lr=5e-2, tol=1e-6))
    solver = ProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    x0 = torch.zeros(2, dim)
    y = torch.randn(2, dim)

    try:
        solver.solve(
            x0=x0,
            y=y,
            lam=0.1,
            max_iter=2,
            differentiable=True,
            line_search=True,
        )
    except ValueError as exc:
        assert "line_search" in str(exc)
    else:
        raise AssertionError("Expected ValueError when line_search=True and differentiable=True")
