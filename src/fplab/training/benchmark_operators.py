from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from fplab.training.train_unrolled import TrainConfig, train_synthetic


@dataclass
class BenchmarkConfig:
    seed: int = 0
    deterministic: bool = False
    dim: int = 16
    batch_size: int = 16
    train_steps: int = 20
    solver_iters: int = 4
    prox_iters: int = 30
    operators: tuple[str, ...] = ("identity", "random", "blur")


def run_benchmark(cfg: BenchmarkConfig) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for idx, operator in enumerate(cfg.operators):
        metrics = train_synthetic(
            TrainConfig(
                seed=cfg.seed + idx,
                deterministic=cfg.deterministic,
                dim=cfg.dim,
                batch_size=cfg.batch_size,
                train_steps=cfg.train_steps,
                solver_iters=cfg.solver_iters,
                prox_iters=cfg.prox_iters,
                operator=operator,
                fixed_batch=True,
            )
        )
        out[operator] = metrics
    return out


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark unrolled training across operators.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--solver-iters", type=int, default=4)
    parser.add_argument("--prox-iters", type=int, default=30)
    parser.add_argument(
        "--operators",
        type=str,
        default="identity,random,blur",
        help="Comma-separated operator list.",
    )
    args = parser.parse_args()

    return BenchmarkConfig(
        seed=args.seed,
        deterministic=args.deterministic,
        dim=args.dim,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        solver_iters=args.solver_iters,
        prox_iters=args.prox_iters,
        operators=tuple(op.strip() for op in args.operators.split(",") if op.strip()),
    )


def main() -> None:
    cfg = _parse_args()
    results = run_benchmark(cfg)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
