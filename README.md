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
