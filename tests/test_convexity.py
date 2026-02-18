from __future__ import annotations

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer


def test_icnn_structural_convexity_constraints() -> None:
    model = ICNNRegularizer(ICNNConfig(input_dim=8, hidden_dims=(16, 16), mu_quadratic=1e-3))
    assert model.convexity_sanity_check()


def test_icnn_convexity_jensen_empirical() -> None:
    torch.manual_seed(0)
    model = ICNNRegularizer(ICNNConfig(input_dim=6, hidden_dims=(12, 12), mu_quadratic=1e-3))

    x = torch.randn(32, 6)
    z = torch.randn(32, 6)
    t = torch.rand(32, 1)

    lhs = model(t * x + (1.0 - t) * z)
    rhs = (t.squeeze(-1) * model(x)) + ((1.0 - t.squeeze(-1)) * model(z))

    # Allow small numerical slack from finite precision.
    assert torch.all(lhs <= rhs + 1e-4)
