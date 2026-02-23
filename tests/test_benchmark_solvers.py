from __future__ import annotations

import math

import pytest
import torch

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
    assert set(results["identity"].keys()) == {"pg", "pg_ls", "fista", "fista_ls", "anderson"}

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
    assert "`fista_ls`" in report
    assert "`anderson`" in report


def test_solver_benchmark_rejects_invalid_config() -> None:
    with pytest.raises(ValueError, match="trials must be >= 1"):
        run_solver_benchmark(SolverBenchmarkConfig(trials=0))


def test_solver_benchmark_image_folder_mode(tmp_path) -> None:
    for idx in range(3):
        sample = torch.rand(8, 8) + 0.1 * idx
        torch.save(sample, tmp_path / f"sample_{idx}.pt")

    results = run_solver_benchmark(
        SolverBenchmarkConfig(
            seed=0,
            dataset="image_folder",
            data_root=str(tmp_path),
            patch_size=4,
            num_images=3,
            batch_size=3,
            iters=4,
            prox_iters=8,
            trials=1,
            operators=("identity",),
        )
    )

    assert set(results.keys()) == {"identity"}
    assert set(results["identity"].keys()) == {"pg", "pg_ls", "fista", "fista_ls", "anderson"}
    for method_metrics in results["identity"].values():
        assert method_metrics["residual_mean"] >= 0.0
        for value in method_metrics.values():
            assert math.isfinite(float(value))


def test_solver_benchmark_image_folder_requires_data_root() -> None:
    with pytest.raises(ValueError, match="data_root is required"):
        run_solver_benchmark(SolverBenchmarkConfig(dataset="image_folder", data_root=None))
