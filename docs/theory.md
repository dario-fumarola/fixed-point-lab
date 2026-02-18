# Theory (MVP)

We target convex composite objectives:

F(x) = f(x; y) + lambda * R_theta(x)

with f convex smooth (Lipschitz gradient) and R_theta proper closed convex.
The solver uses proximal-gradient updates:

x_{k+1} = prox_{alpha * lambda * R_theta}(x_k - alpha * grad f(x_k; y))

For alpha in (0, 1 / L_f], proximal-gradient converges to a minimizer under standard assumptions.
