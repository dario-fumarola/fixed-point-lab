from __future__ import annotations

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig


def test_prox_solution_is_invariant_to_batch_repetition() -> None:
    torch.manual_seed(0)
    dim = 4
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(8, 8), mu_quadratic=1e-2))
    prox = ICNNProxSolver(ProxConfig(max_iters=40, lr=5e-2, tol=1e-8))

    v_single = torch.randn(1, dim)
    x_single, _single_info = prox.prox(
        v_single,
        alpha=0.5,
        lam=0.2,
        regularizer=regularizer,
        differentiable=False,
    )

    v_batched = v_single.repeat(8, 1)
    x_batched, _batch_info = prox.prox(
        v_batched,
        alpha=0.5,
        lam=0.2,
        regularizer=regularizer,
        differentiable=False,
    )

    assert torch.allclose(x_batched[0], x_single[0], atol=1e-5, rtol=1e-5)
