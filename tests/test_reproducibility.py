from __future__ import annotations

import numpy as np
import torch

from fplab.utils.reproducibility import set_seed


def test_set_seed_repeats_torch_and_numpy_streams() -> None:
    set_seed(123)
    a_torch = torch.randn(5)
    a_np = np.random.randn(5)

    set_seed(123)
    b_torch = torch.randn(5)
    b_np = np.random.randn(5)

    assert torch.allclose(a_torch, b_torch)
    assert np.allclose(a_np, b_np)
