# Unrolled Training (MVP)

The unrolled trainer optimizes ICNN regularizer parameters by differentiating through a fixed number of proximal-gradient steps.

For each batch:
1. Sample synthetic inverse problem pairs `(x_true, y)`.
2. Run `K` solver steps with `differentiable=True` and `early_stop=False`.
3. Backpropagate reconstruction + data-consistency losses into ICNN parameters.

This gives a practical training loop while keeping the inference solver aligned with the proximal-gradient fixed-point template.

Guarantee boundary:
- Inference-mode convergence arguments rely on convex assumptions and step-size/safeguard conditions.
- Unrolled training uses finite differentiable iterations and is optimized as a bilevel surrogate objective.
- During differentiable unrolling, some non-smooth safeguards are intentionally disabled to keep gradients well-defined.

Solver choices:
- `pg`: plain proximal-gradient updates.
- `fista`: accelerated proximal-gradient updates.

Use `--solver fista` in training/demo CLIs to enable acceleration.

For inference diagnostics in the synthetic demo, proximal-gradient also supports
backtracking line search via `--line-search` and `--alpha-scale`.

For solver-level comparisons (not training), use:
```bash
fplab-benchmark-solvers --dim 16 --iters 20 --trials 3 --operators identity,random,blur
```
It benchmarks `pg`, `pg_ls` (PG + line search), `fista`, and `anderson`,
then writes a markdown report to `reports/solver_benchmark.md` by default.

To benchmark on local real samples, switch dataset mode:
```bash
fplab-benchmark-solvers \
  --dataset image_folder \
  --data-root /path/to/local/images_or_pt_tensors \
  --patch-size 16 \
  --num-images 64 \
  --iters 20 \
  --trials 3
```
In `image_folder` mode, the benchmark samples patches from local images (or `.pt/.pth`
tensors), flattens them into vectors, and evaluates the same solver/operator variants.

Operator options currently supported:
- `identity`: denoising-style setting.
- `random`: normalized dense sensing matrix.
- `blur`: normalized 1D circular blur operator.
