# Theory (MVP)

We target convex composite objectives:

F(x) = f(x; y) + lambda * R_theta(x)

with f convex smooth (Lipschitz gradient) and R_theta proper closed convex.
The solver uses proximal-gradient updates:

x_{k+1} = prox_{alpha * lambda * R_theta}(x_k - alpha * grad f(x_k; y))

For alpha in (0, 1 / L_f], proximal-gradient converges to a minimizer under standard assumptions.

When an aggressive step estimate is used, a backtracking majorization check can be applied:

f(x_{k+1}) <= f(x_k) + <grad f(x_k), x_{k+1} - x_k> + (1 / (2 alpha_k)) ||x_{k+1} - x_k||^2

This recovers stable descent by shrinking `alpha_k` until the condition is satisfied.

The repo also supports accelerated proximal-gradient (FISTA):

y_k = x_k + beta_k (x_k - x_{k-1})
x_{k+1} = prox_{alpha * lambda * R_theta}(y_k - alpha * grad f(y_k; y))

with optional monotone safeguarding in inference mode (fallback to a plain PG step when acceleration
increases the objective).

For generic fixed-point maps `x = T(x)`, the codebase includes:
- Krasnoselskii-Mann iteration
- Anderson acceleration (regularized multisecant mixing)
