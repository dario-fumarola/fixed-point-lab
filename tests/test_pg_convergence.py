from __future__ import annotations

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.proxgrad import ProxGradSolver


def test_proxgrad_objective_is_nonincreasing_on_quadratic_task() -> None:
    torch.manual_seed(0)

    n = 6
    A = torch.eye(n)
    fidelity = LeastSquaresFidelity(A=A)

    reg = ICNNRegularizer(ICNNConfig(input_dim=n, hidden_dims=(16, 16), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=200, lr=5e-2, tol=1e-6))
    solver = ProxGradSolver(fidelity=fidelity, regularizer=reg, prox_solver=prox)

    x0 = torch.randn(8, n)
    y = torch.randn(8, n)

    _, trace = solver.solve(x0=x0, y=y, lam=0.1, max_iter=20, tol=1e-6)
    objectives = trace.objectives

    assert len(objectives) > 2
    for prev, curr in zip(objectives, objectives[1:]):
        assert curr <= prev + 1e-4
