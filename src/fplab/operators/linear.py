from __future__ import annotations

import torch


def normalize_operator(A: torch.Tensor) -> torch.Tensor:
    """Scale matrix so ||A||_2 <= 1 for stable optimization defaults."""
    spectral = torch.linalg.matrix_norm(A, ord=2)
    return A / torch.clamp(spectral, min=1e-6)


def make_random_operator(dim: int) -> torch.Tensor:
    A = torch.randn(dim, dim) / (dim**0.5)
    return normalize_operator(A)


def make_identity_operator(dim: int) -> torch.Tensor:
    return torch.eye(dim)


def make_blur_operator(dim: int, kernel: torch.Tensor | None = None) -> torch.Tensor:
    """Create a 1D circular blur matrix as a normalized circulant operator."""
    if kernel is None:
        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)

    if kernel.dim() != 1:
        raise ValueError("kernel must be 1D")

    k = kernel / torch.clamp(torch.sum(kernel), min=1e-6)
    first_col = torch.zeros(dim, dtype=torch.float32)
    first_col[: len(k)] = k

    rows = []
    for shift in range(dim):
        rows.append(torch.roll(first_col, shifts=shift))
    A = torch.stack(rows, dim=0)
    return normalize_operator(A)
