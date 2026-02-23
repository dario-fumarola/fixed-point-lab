from __future__ import annotations

from pathlib import Path

from fplab.training.train_unrolled import TrainConfig, train_synthetic


def test_train_unrolled_smoke_fixed_batch_improves(tmp_path: Path) -> None:
    ckpt_path = tmp_path / "ckpt.pt"
    metrics = train_synthetic(
        TrainConfig(
            seed=0,
            dim=8,
            batch_size=8,
            solver_iters=3,
            prox_iters=20,
            train_steps=10,
            learning_rate=2e-3,
            fixed_batch=True,
            operator="blur",
            save_path=str(ckpt_path),
        )
    )

    assert int(metrics["train_steps"]) == 10
    assert float(metrics["best_loss"]) <= float(metrics["initial_loss"]) + 1e-6
    assert float(metrics["final_loss"]) <= float(metrics["initial_loss"]) * 1.10
    assert ckpt_path.exists()


def test_train_unrolled_smoke_with_fista_solver() -> None:
    metrics = train_synthetic(
        TrainConfig(
            seed=0,
            dim=8,
            batch_size=8,
            solver_iters=3,
            prox_iters=20,
            train_steps=6,
            learning_rate=2e-3,
            fixed_batch=True,
            operator="identity",
            solver="fista",
        )
    )

    assert metrics["solver"] == "fista"
    assert float(metrics["best_loss"]) <= float(metrics["initial_loss"]) + 1e-6
