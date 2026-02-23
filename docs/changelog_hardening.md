# Changelog: Hardening + Performance Pass

## Summary
This pass improves correctness, robustness, reproducibility, and selected solver hot-path performance without changing public command names.

## Added
- Regression tests for:
  - ICNN config validation
  - Prox config validation
  - Prox batch-size invariance
  - Blur operator edge cases (`dim=1`, `dim=2`, long kernels)
  - PG line-search fallback finite diagnostics
  - Deterministic mode roundtrip
  - Deterministic file ordering in image pool builder
  - Anderson batched decoupling
  - Stress solver regressions
- New trace field:
  - `line_search_accepted` in `SolveTrace` and `FISTATrace`

## Changed
- ICNN config now validates `input_dim >= 1`, non-empty `hidden_dims`, and positive hidden widths.
- Prox config now validates `max_iters >= 1`, `lr > 0`, and nonnegative tolerances.
- Prox update is now batch-size invariant (sum-based objective in inner solve).
- PG/FISTA line search now:
  - checks majorization per sample,
  - always returns finite candidate diagnostics,
  - reports acceptance per iteration.
- Anderson acceleration in batched mode now decouples samples while using batched linear algebra for speed.
- Least-squares fidelity now caches the Lipschitz constant for fixed `A`.
- Local image pool file discovery now sorts paths for deterministic selection.
- Deterministic mode toggle now explicitly enables/disables deterministic algorithms both ways.
- Documentation now scopes convergence claims by assumptions and mode (inference vs differentiable unrolling).

## Removed
- Unused runtime dependency: `hydra-core`.

## Performance notes
- Microbench (same workload, local CPU):
  - PG + line search loop (`10` runs of solve): `3.09s -> 2.82s` (~`8.9%` faster).
  - Batched Anderson loop (`50` runs): `2.50s -> 0.15s` (large speedup from batched KKT solve path).

## Validation
- `uv run pytest -q`
- `uv run pytest -q tests/test_stress_solver_regressions.py`
- `uv run ruff check`
- CLI smokes:
  - `uv run fplab-demo --dim 16 --iters 8 --solver pg --line-search --alpha-scale 6.0`
  - `uv run fplab-demo --dim 16 --iters 8 --solver fista`
  - `uv run fplab-benchmark-solvers --dim 8 --iters 4 --trials 1 --operators identity`
