from __future__ import annotations

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig


def test_prox_returns_finite_and_decreases_distance_term() -> None:
    torch.manual_seed(0)
    reg = ICNNRegularizer(ICNNConfig(input_dim=5, hidden_dims=(10, 10), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=120, lr=1e-1, tol=1e-5))

    v = torch.randn(16, 5)
    x, info = prox.prox(v, alpha=0.5, lam=0.2, regularizer=reg)

    assert torch.isfinite(x).all()
    assert info.iters >= 1
    assert info.grad_norm >= 0.0
    assert info.threshold >= 0.0

    # Prox of a strongly convex regularizer should not move arbitrarily far away.
    moved = torch.linalg.norm(x - v).item()
    assert moved < 10.0


def test_prox_relative_tolerance_stops_early() -> None:
    torch.manual_seed(0)
    reg = ICNNRegularizer(ICNNConfig(input_dim=4, hidden_dims=(8, 8), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=100, lr=2e-2, tol=1e-12, rel_tol=1e2))

    v = torch.randn(2, 4)
    _x, info = prox.prox(v, alpha=0.5, lam=0.2, regularizer=reg)

    assert info.converged
    assert info.iters < 100
