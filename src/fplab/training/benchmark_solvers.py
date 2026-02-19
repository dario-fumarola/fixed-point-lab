from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter

import torch

from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.operators.linear import make_blur_operator, make_identity_operator, make_random_operator
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig
from fplab.solvers.fista import FISTAProxGradSolver
from fplab.solvers.fixed_point import anderson_acceleration
from fplab.solvers.proxgrad import ProxGradSolver
from fplab.utils.reproducibility import set_seed


@dataclass
class SolverBenchmarkConfig:
    seed: int = 0
    deterministic: bool = False
    dim: int = 16
    batch_size: int = 8
    noise_std: float = 0.03
    lam: float = 0.1
    iters: int = 20
    prox_iters: int = 50
    trials: int = 3
    alpha_scale_ls: float = 6.0
    anderson_history: int = 5
    operators: tuple[str, ...] = ("identity", "random", "blur")
    report_path: str | None = None


_METHODS = ("pg", "pg_ls", "fista", "anderson")


def _validate_cfg(cfg: SolverBenchmarkConfig) -> None:
    if cfg.dim < 1:
        raise ValueError("dim must be >= 1")
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg.iters < 1:
        raise ValueError("iters must be >= 1")
    if cfg.prox_iters < 1:
        raise ValueError("prox_iters must be >= 1")
    if cfg.trials < 1:
        raise ValueError("trials must be >= 1")
    if cfg.alpha_scale_ls <= 0.0:
        raise ValueError("alpha_scale_ls must be > 0")
    if cfg.anderson_history < 1:
        raise ValueError("anderson_history must be >= 1")
    if not cfg.operators:
        raise ValueError("operators must contain at least one entry")


def _make_operator(dim: int, operator: str) -> torch.Tensor:
    if operator == "random":
        return make_random_operator(dim)
    if operator == "identity":
        return make_identity_operator(dim)
    if operator == "blur":
        return make_blur_operator(dim)
    raise ValueError(f"unsupported operator: {operator}")


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


def _psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2)
    peak = torch.max(torch.abs(target)) + eps
    val = 20.0 * torch.log10(peak / torch.sqrt(mse + eps))
    return float(val.item())


def _init_accumulators() -> dict[str, list[float]]:
    return {
        "objective": [],
        "residual": [],
        "iters": [],
        "psnr": [],
        "time_sec": [],
    }


def _aggregate(metric_lists: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, values in metric_lists.items():
        out[f"{key}_mean"] = float(mean(values))
        out[f"{key}_std"] = float(pstdev(values)) if len(values) > 1 else 0.0
    return out


def _run_single_method(
    method: str,
    pg_solver: ProxGradSolver,
    fista_solver: FISTAProxGradSolver,
    x0: torch.Tensor,
    y: torch.Tensor,
    x_true: torch.Tensor,
    lam: float,
    iters: int,
    alpha_scale_ls: float,
    anderson_history: int,
) -> dict[str, float]:
    start = perf_counter()

    if method == "pg":
        x_hat, trace = pg_solver.solve(
            x0=x0,
            y=y,
            lam=lam,
            max_iter=iters,
            tol=1e-6,
            alpha_scale=1.0,
            line_search=False,
        )
        objective = trace.objectives[-1]
        residual = trace.residuals[-1]
        iter_count = float(len(trace.residuals))
    elif method == "pg_ls":
        x_hat, trace = pg_solver.solve(
            x0=x0,
            y=y,
            lam=lam,
            max_iter=iters,
            tol=1e-6,
            alpha_scale=alpha_scale_ls,
            line_search=True,
        )
        objective = trace.objectives[-1]
        residual = trace.residuals[-1]
        iter_count = float(len(trace.residuals))
    elif method == "fista":
        x_hat, trace = fista_solver.solve(
            x0=x0,
            y=y,
            lam=lam,
            max_iter=iters,
            tol=1e-6,
            monotone=True,
            adaptive_restart=True,
        )
        objective = trace.objectives[-1]
        residual = trace.residuals[-1]
        iter_count = float(len(trace.residuals))
    elif method == "anderson":
        Lf = pg_solver.fidelity.lipschitz()
        alpha = 1.0 / Lf

        def pg_operator(x: torch.Tensor) -> torch.Tensor:
            v = x - alpha * pg_solver.fidelity.grad(x, y)
            x_next, _info = pg_solver.prox_solver.prox(
                v=v,
                alpha=alpha,
                lam=lam,
                regularizer=pg_solver.regularizer,
                differentiable=False,
            )
            return x_next

        x_hat, trace = anderson_acceleration(
            operator=pg_operator,
            x0=x0,
            history=anderson_history,
            damping=1.0,
            reg=1e-4,
            max_iter=iters,
            tol=1e-6,
        )
        objective = float(torch.mean(pg_solver.objective(x_hat, y, lam)).item())
        residual = trace.residuals[-1]
        iter_count = float(len(trace.residuals))
    else:
        raise ValueError(f"unsupported method: {method}")

    elapsed = perf_counter() - start
    psnr = _psnr(x_hat, x_true)

    return {
        "objective": float(objective),
        "residual": float(residual),
        "iters": iter_count,
        "psnr": float(psnr),
        "time_sec": float(elapsed),
    }


def _format_markdown_report(
    results: dict[str, dict[str, dict[str, float]]],
    cfg: SolverBenchmarkConfig,
) -> str:
    lines: list[str] = []
    lines.append("# Solver Benchmark Report")
    lines.append("")
    lines.append(f"- dim: `{cfg.dim}`")
    lines.append(f"- batch_size: `{cfg.batch_size}`")
    lines.append(f"- noise_std: `{cfg.noise_std}`")
    lines.append(f"- lambda: `{cfg.lam}`")
    lines.append(f"- iterations: `{cfg.iters}`")
    lines.append(f"- trials: `{cfg.trials}`")
    lines.append("")

    for operator in cfg.operators:
        lines.append(f"## Operator: `{operator}`")
        lines.append("")
        lines.append("| Method | Obj (mean±std) | Residual (mean±std) | PSNR (mean±std) | Iters (mean±std) | Time s (mean±std) |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        for method in _METHODS:
            metrics = results[operator][method]
            lines.append(
                "| "
                + f"`{method}` | "
                + f"{metrics['objective_mean']:.6f} ± {metrics['objective_std']:.6f} | "
                + f"{metrics['residual_mean']:.6f} ± {metrics['residual_std']:.6f} | "
                + f"{metrics['psnr_mean']:.3f} ± {metrics['psnr_std']:.3f} | "
                + f"{metrics['iters_mean']:.2f} ± {metrics['iters_std']:.2f} | "
                + f"{metrics['time_sec_mean']:.4f} ± {metrics['time_sec_std']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_solver_benchmark(cfg: SolverBenchmarkConfig) -> dict[str, dict[str, dict[str, float]]]:
    _validate_cfg(cfg)
    results: dict[str, dict[str, dict[str, float]]] = {}

    for op_idx, operator in enumerate(cfg.operators):
        per_method_lists: dict[str, dict[str, list[float]]] = {m: _init_accumulators() for m in _METHODS}

        for trial in range(cfg.trials):
            set_seed(cfg.seed + 1000 * op_idx + trial, deterministic=cfg.deterministic)

            A = _make_operator(cfg.dim, operator)
            fidelity = LeastSquaresFidelity(A=A)
            regularizer = ICNNRegularizer(
                ICNNConfig(input_dim=cfg.dim, hidden_dims=(64, 64), mu_quadratic=1e-2)
            )
            prox = ICNNProxSolver(ProxConfig(max_iters=cfg.prox_iters, lr=5e-2, tol=1e-6))
            pg_solver = ProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)
            fista_solver = FISTAProxGradSolver(fidelity=fidelity, regularizer=regularizer, prox_solver=prox)

            x_true, y = _sample_batch(cfg.batch_size, cfg.dim, A, cfg.noise_std)
            x0 = torch.zeros_like(x_true)

            for method in _METHODS:
                metrics = _run_single_method(
                    method=method,
                    pg_solver=pg_solver,
                    fista_solver=fista_solver,
                    x0=x0,
                    y=y,
                    x_true=x_true,
                    lam=cfg.lam,
                    iters=cfg.iters,
                    alpha_scale_ls=cfg.alpha_scale_ls,
                    anderson_history=cfg.anderson_history,
                )
                for k, v in metrics.items():
                    per_method_lists[method][k].append(float(v))

        results[operator] = {method: _aggregate(values) for method, values in per_method_lists.items()}

    return results


def write_solver_benchmark_report(
    results: dict[str, dict[str, dict[str, float]]],
    cfg: SolverBenchmarkConfig,
) -> Path:
    report_path = Path(cfg.report_path) if cfg.report_path else Path("reports/solver_benchmark.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_format_markdown_report(results, cfg))
    return report_path


def _parse_args() -> SolverBenchmarkConfig:
    parser = argparse.ArgumentParser(description="Benchmark PG/FISTA/Anderson fixed-point solvers.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--prox-iters", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--alpha-scale-ls", type=float, default=6.0)
    parser.add_argument("--anderson-history", type=int, default=5)
    parser.add_argument(
        "--operators",
        type=str,
        default="identity,random,blur",
        help="Comma-separated operator list.",
    )
    parser.add_argument("--report-path", type=str, default=None)
    args = parser.parse_args()
    operators = tuple(op.strip() for op in args.operators.split(",") if op.strip())
    if not operators:
        parser.error("--operators must contain at least one operator")

    return SolverBenchmarkConfig(
        seed=args.seed,
        deterministic=args.deterministic,
        dim=args.dim,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        lam=args.lam,
        iters=args.iters,
        prox_iters=args.prox_iters,
        trials=args.trials,
        alpha_scale_ls=args.alpha_scale_ls,
        anderson_history=args.anderson_history,
        operators=operators,
        report_path=args.report_path,
    )


def main() -> None:
    cfg = _parse_args()
    results = run_solver_benchmark(cfg)
    report_path = write_solver_benchmark_report(results, cfg)
    payload = {"report_path": str(report_path), "results": results}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
