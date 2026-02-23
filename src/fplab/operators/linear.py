from __future__ import annotations

import torch


def _validate_dim(dim: int) -> None:
    if dim < 1:
        raise ValueError("dim must be >= 1")


def normalize_operator(A: torch.Tensor) -> torch.Tensor:
    """Scale matrix so ||A||_2 <= 1 for stable optimization defaults."""
    spectral = torch.linalg.matrix_norm(A, ord=2)
    return A / torch.clamp(spectral, min=1e-6)


def make_random_operator(dim: int) -> torch.Tensor:
    _validate_dim(dim)
    A = torch.randn(dim, dim) / (dim**0.5)
    return normalize_operator(A)


def make_identity_operator(dim: int) -> torch.Tensor:
    _validate_dim(dim)
    return torch.eye(dim)


def make_blur_operator(dim: int, kernel: torch.Tensor | None = None) -> torch.Tensor:
    """Create a 1D circular blur matrix as a normalized circulant operator."""
    _validate_dim(dim)
    if kernel is None:
        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)

    if kernel.dim() != 1:
        raise ValueError("kernel must be 1D")
    if kernel.numel() == 0:
        raise ValueError("kernel must be non-empty")

    k = kernel.to(dtype=torch.float32)
    k = k / torch.clamp(torch.sum(k), min=1e-6)
    first_col = torch.zeros(dim, dtype=torch.float32)
    for idx, value in enumerate(k):
        first_col[idx % dim] += value

    rows = []
    for shift in range(dim):
        rows.append(torch.roll(first_col, shifts=shift))
    A = torch.stack(rows, dim=0)
    return normalize_operator(A)
