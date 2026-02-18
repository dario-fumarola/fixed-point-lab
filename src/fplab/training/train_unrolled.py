from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.proxgrad import ProxGradSolver


@dataclass
class TrainConfig:
    seed: int = 0
    dim: int = 32
    batch_size: int = 32
    noise_std: float = 0.03
    lam: float = 0.1
    solver_iters: int = 6
    prox_iters: int = 60
    train_steps: int = 80
    learning_rate: float = 1e-3
    grad_clip: float = 5.0
    fixed_batch: bool = False


def _make_operator(dim: int) -> torch.Tensor:
    """Create a normalized square sensing operator with ||A||_2 ~ 1."""
    A = torch.randn(dim, dim) / (dim**0.5)
    smax = torch.linalg.matrix_norm(A, ord=2)
    return A / torch.clamp(smax, min=1e-6)


def _sample_batch(
    batch_size: int,
    dim: int,
    A: torch.Tensor,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_true = torch.randn(batch_size, dim)
    y_clean = x_true @ A.T
    y = y_clean + noise_std * torch.randn_like(y_clean)
    return x_true, y


def train_synthetic(cfg: TrainConfig) -> dict[str, float | int]:
    torch.manual_seed(cfg.seed)

    A = _make_operator(cfg.dim)
    fidelity = LeastSquaresFidelity(A=A)

    regularizer = ICNNRegularizer(
        ICNNConfig(input_dim=cfg.dim, hidden_dims=(64, 64), mu_quadratic=1e-2)
    )
    prox = ICNNProxSolver(ProxConfig(max_iters=cfg.prox_iters, lr=3e-2, tol=1e-6))
    solver = ProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

    optimizer = torch.optim.Adam(regularizer.parameters(), lr=cfg.learning_rate)

    fixed = None
    if cfg.fixed_batch:
        fixed = _sample_batch(cfg.batch_size, cfg.dim, A, cfg.noise_std)

    loss_history: list[float] = []

    for _step in range(cfg.train_steps):
        x_true, y = fixed if fixed is not None else _sample_batch(cfg.batch_size, cfg.dim, A, cfg.noise_std)
        x0 = torch.zeros_like(x_true)

        x_hat, _trace = solver.solve(
            x0=x0,
            y=y,
            lam=cfg.lam,
            max_iter=cfg.solver_iters,
            tol=0.0,
            differentiable=True,
            early_stop=False,
        )

        rec_loss = torch.mean((x_hat - x_true) ** 2)
        data_loss = torch.mean(fidelity.value(x_hat, y))
        # Keep the learned regularizer active while prioritizing reconstruction quality.
        total_loss = rec_loss + 0.2 * data_loss + 1e-4 * torch.mean(regularizer(x_hat))

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(regularizer.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()

        loss_history.append(float(total_loss.detach().item()))

    return {
        "train_steps": cfg.train_steps,
        "initial_loss": loss_history[0],
        "final_loss": loss_history[-1],
        "best_loss": min(loss_history),
    }


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ICNN regularizer with unrolled prox-gradient.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--solver-iters", type=int, default=6)
    parser.add_argument("--prox-iters", type=int, default=60)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--fixed-batch", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        seed=args.seed,
        dim=args.dim,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        lam=args.lam,
        solver_iters=args.solver_iters,
        prox_iters=args.prox_iters,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        grad_clip=args.grad_clip,
        fixed_batch=args.fixed_batch,
    )


def main() -> None:
    cfg = _parse_args()
    metrics = train_synthetic(cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
