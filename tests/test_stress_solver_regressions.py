from __future__ import annotations

import math

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.fista import FISTAProxGradSolver
from fplab.solvers.proxgrad import ProxGradSolver


def _all_finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def test_stress_pg_linesearch_traces_are_finite() -> None:
    for seed in range(3):
        torch.manual_seed(seed)
        dim = 6
        fidelity = LeastSquaresFidelity(A=torch.eye(dim))
        regularizer = ICNNRegularizer(
            ICNNConfig(input_dim=dim, hidden_dims=(12, 12), mu_quadratic=1e-2)
        )
        prox = ICNNProxSolver(ProxConfig(max_iters=25, lr=5e-2, tol=1e-6))
        solver = ProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

        x0 = torch.randn(4, dim)
        y = torch.randn(4, dim)
        _x_hat, trace = solver.solve(
            x0=x0,
            y=y,
            lam=0.1,
            max_iter=8,
            tol=1e-6,
            line_search=True,
            alpha_scale=8.0,
        )

        assert _all_finite(trace.objectives)
        assert _all_finite(trace.residuals)
        assert _all_finite(trace.prox_grad_norms)
        assert _all_finite(trace.prox_thresholds)
        assert _all_finite(trace.step_sizes)
        assert len(trace.line_search_accepted) == len(trace.objectives)


def test_stress_fista_linesearch_traces_are_finite() -> None:
    for seed in range(3):
        torch.manual_seed(seed)
        dim = 6
        fidelity = LeastSquaresFidelity(A=torch.eye(dim))
        regularizer = ICNNRegularizer(
            ICNNConfig(input_dim=dim, hidden_dims=(12, 12), mu_quadratic=1e-2)
        )
        prox = ICNNProxSolver(ProxConfig(max_iters=25, lr=5e-2, tol=1e-6))
        solver = FISTAProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

        x0 = torch.randn(4, dim)
        y = torch.randn(4, dim)
        _x_hat, trace = solver.solve(
            x0=x0,
            y=y,
            lam=0.1,
            max_iter=8,
            tol=1e-6,
            line_search=True,
            alpha_scale=8.0,
            monotone=True,
            adaptive_restart=True,
        )

        assert _all_finite(trace.objectives)
        assert _all_finite(trace.residuals)
        assert _all_finite(trace.prox_grad_norms)
        assert _all_finite(trace.prox_thresholds)
        assert _all_finite(trace.step_sizes)
        assert len(trace.line_search_accepted) == len(trace.objectives)
