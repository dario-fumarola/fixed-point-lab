from __future__ import annotations

import math

import pytest

from fplab.training.benchmark_solvers import (
    SolverBenchmarkConfig,
    run_solver_benchmark,
    write_solver_benchmark_report,
)


def test_solver_benchmark_smoke() -> None:
    results = run_solver_benchmark(
        SolverBenchmarkConfig(
            seed=0,
            dim=8,
            batch_size=4,
            noise_std=0.02,
            lam=0.1,
            iters=6,
            prox_iters=10,
            trials=1,
            operators=("identity",),
        )
    )

    assert set(results.keys()) == {"identity"}
    assert set(results["identity"].keys()) == {"pg", "pg_ls", "fista", "anderson"}

    for method_metrics in results["identity"].values():
        assert method_metrics["residual_mean"] >= 0.0
        assert method_metrics["iters_mean"] >= 1.0
        for value in method_metrics.values():
            assert math.isfinite(float(value))


def test_solver_benchmark_report_written(tmp_path) -> None:
    cfg = SolverBenchmarkConfig(
        seed=0,
        dim=8,
        batch_size=4,
        iters=4,
        prox_iters=8,
        trials=1,
        operators=("identity", "blur"),
        report_path=str(tmp_path / "solver_benchmark.md"),
    )
    results = run_solver_benchmark(cfg)
    report_path = write_solver_benchmark_report(results, cfg)
    report = report_path.read_text()

    assert report_path.exists()
    assert "# Solver Benchmark Report" in report
    assert "## Operator: `identity`" in report
    assert "## Operator: `blur`" in report
    assert "`pg`" in report
    assert "`pg_ls`" in report
    assert "`fista`" in report
    assert "`anderson`" in report


def test_solver_benchmark_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="trials must be >= 1"):
        run_solver_benchmark(SolverBenchmarkConfig(trials=0))
