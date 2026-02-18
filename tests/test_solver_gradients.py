from __future__ import annotations

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.proxgrad import ProxGradSolver


def test_differentiable_solve_backpropagates_into_regularizer() -> None:
    torch.manual_seed(0)

    dim = 6
    A = torch.eye(dim)
    fidelity = LeastSquaresFidelity(A=A)
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(16, 16), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=20, lr=5e-2, tol=1e-7))
    solver = ProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    x_true = torch.randn(4, dim)
    y = x_true + 0.01 * torch.randn_like(x_true)
    x0 = torch.zeros_like(x_true)

    x_hat, _trace = solver.solve(
        x0=x0,
        y=y,
        lam=0.1,
        max_iter=3,
        tol=0.0,
        differentiable=True,
        early_stop=False,
    )
    loss = torch.mean((x_hat - x_true) ** 2)
    loss.backward()

    grad_norms = [
        float(p.grad.norm().item())
        for p in regularizer.parameters()
        if p.grad is not None and torch.isfinite(p.grad).all()
    ]
    assert grad_norms
    assert max(grad_norms) > 0.0
