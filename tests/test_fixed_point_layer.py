from __future__ import annotations

import math

import torch
import pytest

from fplab.layers import FixedPointLayer, FixedPointLayerConfig
from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig


def _build_layer(
    *,
    solver: str = "pg",
    seed: int = 0,
    lam: float = 0.1,
    max_iter: int = 5,
    differentiable: bool = True,
    early_stop: bool = False,
) -> tuple[FixedPointLayer, int]:
    torch.manual_seed(seed)
    dim = 6
    fidelity = LeastSquaresFidelity(A=torch.eye(dim))
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=dim, hidden_dims=(8, 8), mu_quadratic=1e-2))
    prox_solver = ICNNProxSolver(ProxConfig(max_iters=40, lr=5e-2, tol=1e-7))
    cfg = FixedPointLayerConfig(
        solver=solver,
        max_iter=max_iter,
        differentiable=differentiable,
        early_stop=early_stop,
        monotone=False if solver == "fista" else False,
    )
    return (
        FixedPointLayer(
            fidelity=fidelity,
            regularizer=regularizer,
            prox_solver=prox_solver,
            config=cfg,
            lam=lam,
        ),
        dim,
    )


def test_layer_forward_returns_trace_and_runs() -> None:
    layer, dim = _build_layer(max_iter=4)
    y = torch.randn(4, dim)

    x_hat, trace = layer(y, return_trace=True)

    assert x_hat.shape == y.shape
    assert len(trace.residuals) >= 1
    assert len(trace.objectives) == len(trace.residuals)
    assert len(trace.prox_grad_norms) == len(trace.residuals)


def test_layer_default_lam_and_custom_x0_run() -> None:
    layer, dim = _build_layer(max_iter=3, lam=0.05)
    y = torch.randn(3, dim)
    x0 = torch.full_like(y, 0.25)

    x_default = layer(y)
    x_custom = layer(y, x0=x0)

    assert x_default.shape == y.shape
    assert x_custom.shape == y.shape
    assert not torch.allclose(x_custom, torch.zeros_like(x0))


def test_layer_backpropagates_gradients_to_inputs_and_regularizer() -> None:
    layer, dim = _build_layer(max_iter=4, differentiable=True)
    y = torch.randn(5, dim, requires_grad=True)

    x_hat = layer(y)
    loss = torch.mean((x_hat - y) ** 2)
    loss.backward()

    assert y.grad is not None
    assert torch.isfinite(y.grad).all()
    assert y.grad.abs().sum() > 0

    reg_grad_norm = 0.0
    for param in layer.regularizer.parameters():
        if param.grad is not None:
            reg_grad_norm += float(param.grad.abs().sum().item())
    assert reg_grad_norm > 0.0


def test_layer_directional_derivative_matches_autograd() -> None:
    layer, dim = _build_layer(max_iter=6, differentiable=True)
    y = torch.randn(2, dim)
    direction = torch.randn_like(y)

    y.requires_grad_(True)
    x_hat, _ = layer(y, return_trace=True)
    scalar = torch.sum(x_hat)
    (grad,) = torch.autograd.grad(scalar, y)

    _, jvp_val = torch.autograd.functional.jvp(lambda inp: torch.sum(layer(inp)), (y,), (direction,), create_graph=False)

    assert isinstance(jvp_val, torch.Tensor)
    projection = float((grad * direction).sum().item())
    finite_diff = float(jvp_val.item())

    # First-order derivative agreement on a tiny step.
    assert math.isfinite(finite_diff)
    assert math.isfinite(projection)
    assert math.isclose(finite_diff, projection, rel_tol=2e-2, abs_tol=5e-3)


def test_layer_with_fista_solver_runs_and_tracks_restarts() -> None:
    layer, dim = _build_layer(solver="fista", max_iter=4, differentiable=False)
    y = torch.randn(2, dim)

    x_hat, trace = layer(y, return_trace=True)

    assert x_hat.shape == y.shape
    assert len(trace.restarts) == len(trace.objectives)
    assert len(trace.momenta) == len(trace.objectives)
    assert all(isinstance(item, bool) for item in trace.restarts)


def test_invalid_solver_raises() -> None:
    torch.manual_seed(0)
    fidelity = LeastSquaresFidelity(A=torch.eye(2))
    regularizer = ICNNRegularizer(ICNNConfig(input_dim=2, hidden_dims=(8, 8), mu_quadratic=1e-2))
    prox_solver = ICNNProxSolver(ProxConfig(max_iters=10, lr=5e-2))

    with pytest.raises(ValueError, match="unsupported solver"):
        FixedPointLayer(fidelity, regularizer, prox_solver, config=FixedPointLayerConfig(solver="does-not-exist"))


def test_layer_overrides_default_lam_and_runtime_kwargs() -> None:
    layer, dim = _build_layer(max_iter=4, lam=0.2)
    y = torch.randn(3, dim)

    x_default = layer(y)
    x_strong = layer(y, lam=0.8)
    x_fewer_steps, trace = layer(y, solver_overrides={"max_iter": 1}, return_trace=True)

    assert x_default.shape == x_strong.shape == x_fewer_steps.shape
    assert not torch.allclose(x_default, x_strong)
    assert len(trace.residuals) == 1


def test_invalid_solver_override_raises() -> None:
    layer, dim = _build_layer()
    y = torch.randn(2, dim)

    with pytest.raises(ValueError, match="unsupported override"):
        layer(y, solver_overrides={"not_a_solver_arg": 1.0})
