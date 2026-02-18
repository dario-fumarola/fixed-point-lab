from __future__ import annotations

import torch

from fplab.operators.linear import make_blur_operator, make_identity_operator, make_random_operator


def test_identity_operator_is_exact_identity() -> None:
    A = make_identity_operator(6)
    assert torch.allclose(A, torch.eye(6))


def test_random_operator_is_normalized() -> None:
    torch.manual_seed(0)
    A = make_random_operator(10)
    smax = float(torch.linalg.matrix_norm(A, ord=2).item())
    assert smax <= 1.0 + 1e-5


def test_blur_operator_is_square_and_normalized() -> None:
    A = make_blur_operator(12)
    assert A.shape == (12, 12)
    smax = float(torch.linalg.matrix_norm(A, ord=2).item())
    assert smax <= 1.0 + 1e-5
