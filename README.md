# fixed-point-lab

Research repo for convergence-guaranteed neural fixed-point solvers for inverse problems.

## Initial scope
- Convex ICNN regularizer with hard convexity constraints
- Proximal-gradient fixed-point solver
- Generic Krasnoselskii-Mann fixed-point iterator
- Unit tests checking convexity, prox residual behavior, and convergence on least-squares tasks

## Quickstart
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

## Run synthetic demo
```bash
source .venv/bin/activate
fplab-demo --dim 32 --iters 30 --lam 0.1
```

Try accelerated fixed-point updates:
```bash
fplab-demo --dim 32 --iters 30 --lam 0.1 --solver fista
```

Try robust proximal-gradient with backtracking from an aggressive initial step:
```bash
fplab-demo --dim 32 --iters 30 --solver pg --alpha-scale 6.0 --line-search
```

## Train unrolled synthetic model
```bash
source .venv/bin/activate
fplab-train-synth --dim 32 --operator blur --solver-iters 6 --prox-iters 60 --train-steps 80
```

Train with accelerated iterations:
```bash
fplab-train-synth --dim 32 --operator blur --solver fista --solver-iters 6 --prox-iters 60 --train-steps 80
```

Use `--fixed-batch` to quickly sanity-check that the optimization loop can overfit one batch.
Use `--save-path checkpoints/run.pt` to save learned parameters and run metadata.
Use `--deterministic` for stricter reproducibility.

## Benchmark operators
```bash
source .venv/bin/activate
fplab-benchmark-ops --dim 16 --train-steps 20 --operators identity,random,blur
```

Benchmark with the accelerated solver:
```bash
fplab-benchmark-ops --dim 16 --train-steps 20 --solver fista --operators identity,random,blur
```

## Benchmark solver variants
```bash
source .venv/bin/activate
fplab-benchmark-solvers --dim 16 --iters 20 --trials 3 --operators identity,random,blur
```

This writes a markdown table report to `reports/solver_benchmark.md` by default.
Use `--report-path <path>` to choose a different output file.

Run against local real samples (image-folder mode):
```bash
fplab-benchmark-solvers \
  --dataset image_folder \
  --data-root /path/to/local/images_or_pt_tensors \
  --patch-size 16 \
  --num-images 64 \
  --iters 20 \
  --trials 3 \
  --operators identity,random,blur
```

`image_folder` mode accepts common image files (`png/jpg/...`) and `.pt/.pth` tensors.
It samples random patches, flattens them, and applies the same linear inverse operators.

## Differentiable Fixed-Point Layer

Use the new `FixedPointLayer` module for NN-style integration of the solver.

```python
import torch
from fplab.layers import FixedPointLayer, FixedPointLayerConfig
from fplab.models.icnn import ICNNConfig, ICNNRegularizer
from fplab.operators.fidelity import LeastSquaresFidelity
from fplab.prox.prox_icnn import ICNNProxSolver, ProxConfig

layer = FixedPointLayer(
    fidelity=LeastSquaresFidelity(A=torch.eye(6)),
    regularizer=ICNNRegularizer(ICNNConfig(input_dim=6, hidden_dims=(16, 16), mu_quadratic=1e-2)),
    prox_solver=ICNNProxSolver(ProxConfig(max_iters=20, lr=5e-2)),
    config=FixedPointLayerConfig(solver="pg", max_iter=6, differentiable=True),
    lam=0.1,
)

y = torch.randn(4, 6)
x_hat = layer(y)
```

`FixedPointLayer(..., return_trace=True)` returns `(x_hat, trace)` with per-iteration diagnostics.
