from __future__ import annotations

from dataclasses import dataclass

import torch

from fplab.models.icnn import ICNNRegularizer
from fplab.operators.fidelity import Fidelity
from fplab.prox.prox_icnn import ICNNProxSolver


@dataclass
class SolveTrace:
    residuals: list[float]
    objectives: list[float]
    prox_grad_norms: list[float]
    prox_thresholds: list[float]


class ProxGradSolver:
    """Proximal-gradient fixed-point solver with alpha=1/L_f."""

    def __init__(self, fidelity: Fidelity, regularizer: ICNNRegularizer, prox_solver: ICNNProxSolver) -> None:
        self.fidelity = fidelity
        self.regularizer = regularizer
        self.prox_solver = prox_solver

    def step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        lam: float,
        differentiable: bool = False,
    ) -> tuple[torch.Tensor, float, float]:
        if lam < 0:
            raise ValueError("lam must be nonnegative")

        Lf = self.fidelity.lipschitz()
        if Lf <= 0:
            raise ValueError("fidelity lipschitz constant must be positive")

        alpha = 1.0 / Lf
        v = x - alpha * self.fidelity.grad(x, y)
        x_next, info = self.prox_solver.prox(
            v,
            alpha=alpha,
            lam=lam,
            regularizer=self.regularizer,
            differentiable=differentiable,
        )
        return x_next, info.grad_norm, info.threshold

    def objective(self, x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
        return self.fidelity.value(x, y) + lam * self.regularizer(x)

    def solve(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        lam: float,
        max_iter: int = 100,
        tol: float = 1e-5,
        differentiable: bool = False,
        early_stop: bool = True,
    ) -> tuple[torch.Tensor, SolveTrace]:
        x = x0.clone()
        residuals: list[float] = []
        objectives: list[float] = []
        prox_grad_norms: list[float] = []
        prox_thresholds: list[float] = []

        for _ in range(max_iter):
            x_next, prox_grad_norm, prox_threshold = self.step(x, y, lam, differentiable=differentiable)
            residual = float(torch.linalg.norm(x_next - x).item())
            residuals.append(residual)
            prox_grad_norms.append(prox_grad_norm)
            prox_thresholds.append(prox_threshold)
            obj = torch.mean(self.objective(x_next, y, lam))
            objectives.append(float(obj.item()))
            x = x_next

            if early_stop and residual <= tol and prox_grad_norm <= max(tol, 1e-6):
                break

        return x, SolveTrace(
            residuals=residuals,
            objectives=objectives,
            prox_grad_norms=prox_grad_norms,
            prox_thresholds=prox_thresholds,
        )
