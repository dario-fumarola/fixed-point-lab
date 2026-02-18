from __future__ import annotations

from fplab.training.train_unrolled import TrainConfig, train_synthetic


def test_train_unrolled_smoke_fixed_batch_improves() -> None:
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
        )
    )

    assert int(metrics["train_steps"]) == 10
    assert float(metrics["best_loss"]) <= float(metrics["initial_loss"]) + 1e-6
    assert float(metrics["final_loss"]) <= float(metrics["initial_loss"]) * 1.10
