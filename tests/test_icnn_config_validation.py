from __future__ import annotations

import pytest

from fplab.models.icnn import ICNNConfig, ICNNRegularizer


def test_icnn_rejects_nonpositive_input_dim() -> None:
    with pytest.raises(ValueError, match="input_dim must be >= 1"):
        ICNNRegularizer(ICNNConfig(input_dim=0, hidden_dims=(8, 8), mu_quadratic=1e-2))


def test_icnn_rejects_empty_hidden_dims() -> None:
    with pytest.raises(ValueError, match="hidden_dims must be non-empty"):
        ICNNRegularizer(ICNNConfig(input_dim=4, hidden_dims=(), mu_quadratic=1e-2))


def test_icnn_rejects_nonpositive_hidden_width() -> None:
    with pytest.raises(ValueError, match="all hidden_dims entries must be >= 1"):
        ICNNRegularizer(ICNNConfig(input_dim=4, hidden_dims=(8, 0), mu_quadratic=1e-2))
