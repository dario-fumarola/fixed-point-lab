from __future__ import annotations

import torch
import pytest

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.fista import FISTAProxGradSolver


def test_fista_objective_is_nonincreasing_with_monotone_safeguard() -> None:
    torch.manual_seed(0)

    dim = 6
    A = torch.eye(dim)
    fidelity = LeastSquaresFidelity(A=A)
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(16, 16), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=200, lr=5e-2, tol=1e-6))
    solver = FISTAProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    x0 = torch.randn(8, dim)
    y = torch.randn(8, dim)
    _, trace = solver.solve(x0=x0, y=y, lam=0.1, max_iter=25, tol=1e-6, monotone=True)

    assert len(trace.objectives) > 2
    for prev, curr in zip(trace.objectives, trace.objectives[1:]):
        assert curr <= prev + 1e-4

    assert len(trace.restarts) == len(trace.objectives)
    assert len(trace.momenta) == len(trace.objectives)


def test_fista_line_search_uses_backtracking_for_aggressive_step() -> None:
    torch.manual_seed(0)

    dim = 1
    A = torch.tensor([[3.0]])
    fidelity = LeastSquaresFidelity(A=A)
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(16, 16), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=250, lr=5e-2, tol=1e-6))
    solver = FISTAProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    x0 = torch.tensor([[2.0]])
    y = torch.tensor([[1.0]])
    _, trace = solver.solve(
        x0=x0,
        y=y,
        lam=0.1,
        max_iter=6,
        tol=1e-8,
        line_search=True,
        alpha_scale=12.0,
    )

    assert len(trace.backtracks) == len(trace.objectives)
    assert any(backtrack > 0 for backtrack in trace.backtracks)
    assert all(size > 0.0 for size in trace.step_sizes)
    assert all(item >= 0 for item in trace.backtracks)
    assert all(curr <= prev + 1e-4 for prev, curr in zip(trace.objectives, trace.objectives[1:]))


def test_fista_line_search_rejected_in_differentiable_mode() -> None:
    torch.manual_seed(0)

    dim = 3
    fidelity = LeastSquaresFidelity(A=torch.eye(dim))
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(16, 16), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig())
    solver = FISTAProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    x0 = torch.randn(2, dim)
    y = torch.randn(2, dim)

    with pytest.raises(ValueError, match="line_search is only supported with differentiable=False"):
        solver.solve(x0=x0, y=y, lam=0.1, max_iter=2, line_search=True, differentiable=True)
