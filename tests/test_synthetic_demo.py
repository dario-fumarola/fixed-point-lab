from __future__ import annotations

from fplab.training.synthetic_demo import DemoConfig, run_demo


def test_synthetic_demo_runs_and_returns_metrics() -> None:
    metrics = run_demo(DemoConfig(seed=0, batch_size=4, dim=8, iters=8))

    assert int(metrics["iters_ran"]) >= 1
    assert bool(metrics["objective_monotone"])
    assert float(metrics["objective_end"]) <= float(metrics["objective_start"]) + 1e-4


def test_synthetic_demo_runs_with_fista_solver() -> None:
    metrics = run_demo(DemoConfig(seed=0, batch_size=4, dim=8, iters=8, solver="fista"))

    assert int(metrics["iters_ran"]) >= 1
    assert bool(metrics["objective_monotone"])
    assert metrics["solver"] == "fista"


def test_synthetic_demo_runs_with_pg_line_search() -> None:
    metrics = run_demo(
        DemoConfig(seed=0, batch_size=4, dim=8, iters=8, solver="pg", alpha_scale=6.0, line_search=True)
    )

    assert int(metrics["iters_ran"]) >= 1
    assert bool(metrics["objective_monotone"])
    assert metrics["solver"] == "pg"
    assert bool(metrics["line_search"])
