# Assumptions-to-Code Mapping

- Convex R_theta: enforced via ICNN parameterization with nonnegative z-path weights.
- Prox well-posedness: ensured by quadratic term mu/2 * ||x||^2 with mu > 0.
- Valid step size: alpha = 1 / L_f where L_f = ||A||_2^2 for least-squares fidelity.
- Optional robustness: backtracking line search enforces the smoothness majorization condition for `f`.
- Inexact prox stopping: abs+relative threshold on prox gradient norm.
- Convergence diagnostics: objective trace, fixed-point residuals, and prox-gradient traces.
