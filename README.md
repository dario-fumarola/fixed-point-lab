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
