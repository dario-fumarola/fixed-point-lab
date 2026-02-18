# Unrolled Training (MVP)

The unrolled trainer optimizes ICNN regularizer parameters by differentiating through a fixed number of proximal-gradient steps.

For each batch:
1. Sample synthetic inverse problem pairs `(x_true, y)`.
2. Run `K` solver steps with `differentiable=True` and `early_stop=False`.
3. Backpropagate reconstruction + data-consistency losses into ICNN parameters.

This gives a practical training loop while keeping the inference solver aligned with the proximal-gradient fixed-point template.

Operator options currently supported:
- `identity`: denoising-style setting.
- `random`: normalized dense sensing matrix.
- `blur`: normalized 1D circular blur operator.
