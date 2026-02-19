from __future__ import annotations

import torch

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
