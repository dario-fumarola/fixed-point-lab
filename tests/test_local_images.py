from __future__ import annotations

import pytest
import torch

from fplab.data.local_images import build_image_pool, sample_image_patches


def test_build_image_pool_reads_pt_tensors(tmp_path) -> None:
    torch.save(torch.rand(6, 7), tmp_path / "a.pt")
    torch.save(torch.rand(5, 8), tmp_path / "b.pth")

    images = build_image_pool(tmp_path, max_images=2)

    assert len(images) == 2
    assert images[0].ndim == 2
    assert images[1].ndim == 2


def test_sample_image_patches_resizes_small_inputs() -> None:
    images = [torch.rand(3, 3), torch.rand(5, 4)]

    x = sample_image_patches(images=images, batch_size=4, patch_size=4)

    assert x.shape == (4, 16)


def test_build_image_pool_rejects_empty_dir(tmp_path) -> None:
    with pytest.raises(ValueError, match="no supported image/tensor files found"):
        build_image_pool(tmp_path, max_images=4)
