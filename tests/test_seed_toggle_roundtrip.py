from __future__ import annotations

import torch

from fplab.utils.reproducibility import set_seed


def test_set_seed_roundtrip_toggles_deterministic_algorithms() -> None:
    set_seed(123, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()

    set_seed(123, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
