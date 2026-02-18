# fixed-point-lab

Research repo for convergence-guaranteed neural fixed-point solvers for inverse problems.

## Initial scope
- Convex ICNN regularizer with hard convexity constraints
- Proximal-gradient fixed-point solver
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

## Train unrolled synthetic model
```bash
source .venv/bin/activate
fplab-train-synth --dim 32 --operator blur --solver-iters 6 --prox-iters 60 --train-steps 80
```

Use `--fixed-batch` to quickly sanity-check that the optimization loop can overfit one batch.
Use `--save-path checkpoints/run.pt` to save learned parameters and run metadata.
