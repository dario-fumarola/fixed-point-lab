from __future__ import annotations

from fplab.training.benchmark_operators import BenchmarkConfig, run_benchmark


def test_benchmark_operators_smoke() -> None:
    results = run_benchmark(
        BenchmarkConfig(
            seed=0,
            dim=8,
            batch_size=8,
            train_steps=4,
            solver_iters=2,
            prox_iters=10,
            operators=("identity", "blur"),
        )
    )

    assert set(results.keys()) == {"identity", "blur"}
    for metrics in results.values():
        assert float(metrics["initial_loss"]) > 0
        assert float(metrics["best_loss"]) <= float(metrics["initial_loss"]) + 1e-6
