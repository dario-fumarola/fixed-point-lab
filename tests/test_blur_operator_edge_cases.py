from __future__ import annotations

import torch

from fplab.operators.linear import make_blur_operator


def test_blur_operator_supports_small_dimensions() -> None:
    for dim in (1, 2):
        A = make_blur_operator(dim)
        assert A.shape == (dim, dim)
        assert torch.isfinite(A).all()
        smax = float(torch.linalg.matrix_norm(A, ord=2).item())
        assert smax <= 1.0 + 1e-5


def test_blur_operator_wraps_long_kernel() -> None:
    kernel = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    A = make_blur_operator(2, kernel=kernel)
    assert A.shape == (2, 2)
    assert torch.isfinite(A).all()
