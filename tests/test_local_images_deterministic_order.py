from __future__ import annotations

from pathlib import Path

import torch

from fplab.data.local_images import build_image_pool


def test_build_image_pool_sorts_paths_before_truncation(tmp_path, monkeypatch) -> None:
    tensor_a = tmp_path / "a.pt"
    tensor_b = tmp_path / "b.pt"
    tensor_c = tmp_path / "c.pt"
    torch.save(torch.full((2, 2), 1.0), tensor_a)
    torch.save(torch.full((2, 2), 2.0), tensor_b)
    torch.save(torch.full((2, 2), 3.0), tensor_c)

    unsorted_paths = [tensor_c, tensor_a, tensor_b]

    def fake_rglob(self: Path, pattern: str):
        if self == tmp_path and pattern == "*":
            return iter(unsorted_paths)
        return iter(())

    monkeypatch.setattr(Path, "rglob", fake_rglob)
    images = build_image_pool(tmp_path, max_images=2)

    assert len(images) == 2
    assert float(images[0].mean().item()) == 1.0
    assert float(images[1].mean().item()) == 2.0
