from __future__ import annotations

import torch

from fplab.solvers.fixed_point import anderson_acceleration


def test_anderson_batched_matches_independent_runs() -> None:
    torch.manual_seed(0)
    batch_size = 3
    dim = 4
    b = torch.randn(batch_size, dim)
    B = 0.6 * torch.eye(dim)

    def operator_batched(x: torch.Tensor) -> torch.Tensor:
        return x @ B.T + b

    x0_batched = torch.zeros(batch_size, dim)
    x_batched, _trace_batched = anderson_acceleration(
        operator=operator_batched,
        x0=x0_batched,
        history=4,
        max_iter=25,
        tol=1e-12,
    )

    x_independent = []
    for batch_idx in range(batch_size):
        b_i = b[batch_idx]

        def operator_single(x: torch.Tensor, b_single: torch.Tensor = b_i) -> torch.Tensor:
            return x @ B.T + b_single

        x_i, _trace_i = anderson_acceleration(
            operator=operator_single,
            x0=torch.zeros(dim),
            history=4,
            max_iter=25,
            tol=1e-12,
        )
        x_independent.append(x_i)

    x_independent_stacked = torch.stack(x_independent, dim=0)
    assert torch.allclose(x_batched, x_independent_stacked, atol=1e-5, rtol=1e-5)
