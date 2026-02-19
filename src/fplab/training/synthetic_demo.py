from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.fista import FISTAProxGradSolver
from fplab.solvers.proxgrad import ProxGradSolver
from fplab.utils.reproducibility import set_seed


@dataclass
class DemoConfig:
    seed: int = 0
    deterministic: bool = False
    batch_size: int = 8
    dim: int = 32
    noise_std: float = 0.02
    lam: float = 0.1
    iters: int = 30
    solver: str = "pg"


def _psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2)
    peak = torch.max(torch.abs(target)) + eps
    val = 20.0 * torch.log10(peak / torch.sqrt(mse + eps))
    return float(val.item())


def run_demo(cfg: DemoConfig) -> dict[str, float | int | bool]:
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    A = torch.eye(cfg.dim)
    fidelity = LeastSquaresFidelity(A=A)

    reg = ICNNRegularizer(
        ICNNConfig(input_dim=cfg.dim, hidden_dims=(64, 64), mu_quadratic=1e-2)
    )
    prox = ICNNProxSolver(ProxConfig(max_iters=250, lr=5e-2, tol=1e-6))
    if cfg.solver == "fista":
        solver = FISTAProxGradSolver(fidelity=fidelity, regularizer=reg, prox_solver=prox)
    elif cfg.solver == "pg":
        solver = ProxGradSolver(fidelity=fidelity, regularizer=reg, prox_solver=prox)
    else:
        raise ValueError(f"unsupported solver: {cfg.solver}")

    x_true = torch.randn(cfg.batch_size, cfg.dim)
    y = x_true + cfg.noise_std * torch.randn_like(x_true)
    x0 = torch.zeros_like(x_true)

    x_hat, trace = solver.solve(x0=x0, y=y, lam=cfg.lam, max_iter=cfg.iters, tol=1e-6)
    psnr_y = _psnr(y, x_true)
    psnr_x = _psnr(x_hat, x_true)

    monotone = all(
        curr <= prev + 1e-4 for prev, curr in zip(trace.objectives, trace.objectives[1:])
    )

    return {
        "iters_ran": len(trace.objectives),
        "objective_start": trace.objectives[0],
        "objective_end": trace.objectives[-1],
        "residual_end": trace.residuals[-1],
        "objective_monotone": monotone,
        "solver": cfg.solver,
        "psnr_noisy": psnr_y,
        "psnr_recon": psnr_x,
    }


def _parse_args() -> DemoConfig:
    parser = argparse.ArgumentParser(description="Run synthetic fixed-point solver demo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--solver", type=str, default="pg", choices=["pg", "fista"])
    args = parser.parse_args()

    return DemoConfig(
        seed=args.seed,
        deterministic=args.deterministic,
        batch_size=args.batch_size,
        dim=args.dim,
        noise_std=args.noise_std,
        lam=args.lam,
        iters=args.iters,
        solver=args.solver,
    )


def main() -> None:
    cfg = _parse_args()
    metrics = run_demo(cfg)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
