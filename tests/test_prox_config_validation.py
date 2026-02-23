from __future__ import annotations

import pytest

from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig


def test_prox_rejects_zero_max_iters() -> None:
    with pytest.raises(ValueError, match="max_iters must be >= 1"):
        ICNNProxSolver(ProxConfig(max_iters=0))


def test_prox_rejects_nonpositive_lr() -> None:
    with pytest.raises(ValueError, match="lr must be positive"):
        ICNNProxSolver(ProxConfig(lr=0.0))


def test_prox_rejects_negative_abs_tolerance() -> None:
    with pytest.raises(ValueError, match="tol must be nonnegative"):
        ICNNProxSolver(ProxConfig(tol=-1.0))


def test_prox_rejects_negative_relative_tolerance() -> None:
    with pytest.raises(ValueError, match="rel_tol must be nonnegative"):
        ICNNProxSolver(ProxConfig(rel_tol=-1.0))
