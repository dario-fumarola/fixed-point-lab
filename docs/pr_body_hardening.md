## Title
Hardening pass: correctness fixes, deterministic behavior, solver diagnostics, and hot-path speedups

## What this PR does
- Fixes multiple edge-case crashes and reliability gaps.
- Tightens configuration validation across ICNN and prox components.
- Makes prox inner updates batch-size invariant.
- Improves line-search failure handling and adds explicit acceptance diagnostics.
- Refines batched Anderson acceleration to avoid cross-sample coupling and improve performance.
- Improves deterministic behavior for seed toggling and image-file discovery.
- Clarifies convergence claims in docs and removes an unused dependency.

## Key code changes
- `src/fplab/models/icnn.py`
  - Validate `input_dim`, `hidden_dims` non-empty, positive widths.
- `src/fplab/prox/prox_icnn.py`
  - Validate `ProxConfig`.
  - Use sum-based prox objective (batch-size invariant updates).
  - Optimize threshold computation path for `rel_tol == 0`.
- `src/fplab/prox/stopping.py`
  - Clarify `ProxStopInfo` semantics.
- `src/fplab/solvers/proxgrad.py`
  - Per-sample line-search majorization.
  - Finite fallback diagnostics on rejection.
  - Add `line_search_accepted` to `SolveTrace`.
- `src/fplab/solvers/fista.py`
  - Per-sample line-search majorization.
  - Add `line_search_accepted` to `FISTATrace`.
- `src/fplab/solvers/fixed_point.py`
  - Batched per-sample Anderson KKT solves via batched linear algebra.
- `src/fplab/operators/linear.py`
  - Dimension validation and small-dim-safe blur kernel wrapping.
- `src/fplab/operators/fidelity.py`
  - Cache least-squares Lipschitz constant.
- `src/fplab/utils/reproducibility.py`
  - Reversible deterministic algorithm toggling.
- `src/fplab/data/local_images.py`
  - Deterministic sorted path discovery.
- Docs and metadata
  - `README.md`, `docs/theory.md`, `docs/training.md`
  - `pyproject.toml` (remove `hydra-core`)

## Risk / impact notes
- Numeric behavior can shift from previous runs because prox updates are now batch-size invariant and line-search checks are stricter per sample.
- New trace fields are additive:
  - `SolveTrace.line_search_accepted`
  - `FISTATrace.line_search_accepted`
- Dependency surface shrinks by removing unused `hydra-core`.

## Validation performed
- `uv run pytest -q` (full suite): pass
- `uv run pytest -q tests/test_stress_solver_regressions.py`: pass
- `uv run ruff check`: pass
- CLI smokes: pass
  - `uv run fplab-demo --dim 16 --iters 8 --solver pg --line-search --alpha-scale 6.0`
  - `uv run fplab-demo --dim 16 --iters 8 --solver fista`
  - `uv run fplab-benchmark-solvers --dim 8 --iters 4 --trials 1 --operators identity`

## Benchmark snapshots (local CPU)
- PG + line search microbench (`10` runs): `3.09s -> 2.82s` (~`8.9%` faster)
- Batched Anderson microbench (`50` runs): `2.50s -> 0.15s` (substantial speedup)
