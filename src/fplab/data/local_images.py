from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_image_grayscale(path: Path) -> torch.Tensor:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime env if missing.
        raise RuntimeError(
            "Pillow is required for image-folder benchmarking. Install with `uv pip install pillow`."
        ) from exc

    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32) / 255.0
    if arr.ndim != 2:
        raise ValueError(f"expected grayscale 2D image, got shape={arr.shape} for {path}")
    return torch.from_numpy(arr)


def _load_tensor_grayscale(path: Path) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"expected Tensor in {path}, got {type(tensor).__name__}")

    if tensor.ndim == 2:
        out = tensor
    elif tensor.ndim == 3:
        # Support CHW/1HW inputs by averaging channels into grayscale.
        out = tensor.mean(dim=0)
    else:
        raise ValueError(f"unsupported tensor shape {tuple(tensor.shape)} in {path}")

    out = out.float()
    min_val = torch.min(out)
    max_val = torch.max(out)
    if max_val > min_val:
        out = (out - min_val) / (max_val - min_val)
    return out


def _is_supported_path(path: Path) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    return suffix in _IMAGE_EXTENSIONS or suffix in {".pt", ".pth"}


def _load_as_grayscale(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return _load_tensor_grayscale(path)
    return _load_image_grayscale(path)


def build_image_pool(data_root: Path, max_images: int = 64) -> list[torch.Tensor]:
    if max_images < 1:
        raise ValueError("max_images must be >= 1")
    if not data_root.exists():
        raise ValueError(f"data_root does not exist: {data_root}")
    if not data_root.is_dir():
        raise ValueError(f"data_root must be a directory: {data_root}")

    selected: list[Path] = []
    for path in sorted(data_root.rglob("*")):
        if _is_supported_path(path):
            selected.append(path)
            if len(selected) >= max_images:
                break

    if not selected:
        raise ValueError(f"no supported image/tensor files found under: {data_root}")

    images: list[torch.Tensor] = []
    for path in selected:
        image = _load_as_grayscale(path)
        if image.ndim != 2:
            raise ValueError(f"loaded non-2D sample from {path}")
        images.append(image)

    return images


def _ensure_min_size(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")

    h, w = image.shape
    if h >= patch_size and w >= patch_size:
        return image

    out_h = max(h, patch_size)
    out_w = max(w, patch_size)
    resized = F.interpolate(
        image.unsqueeze(0).unsqueeze(0),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0]


def sample_image_patches(
    images: list[torch.Tensor],
    batch_size: int,
    patch_size: int,
) -> torch.Tensor:
    if not images:
        raise ValueError("images must be non-empty")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if patch_size < 1:
        raise ValueError("patch_size must be >= 1")

    vectors: list[torch.Tensor] = []
    image_indices = torch.randint(low=0, high=len(images), size=(batch_size,))

    for img_idx in image_indices.tolist():
        image = _ensure_min_size(images[img_idx], patch_size=patch_size)
        h, w = image.shape

        top = int(torch.randint(low=0, high=h - patch_size + 1, size=(1,)).item())
        left = int(torch.randint(low=0, high=w - patch_size + 1, size=(1,)).item())
        patch = image[top : top + patch_size, left : left + patch_size]
        vectors.append(patch.reshape(-1).float())

    return torch.stack(vectors, dim=0)
