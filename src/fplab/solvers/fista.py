from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fplab.models.icnn import ICNNRegularizer
from fplab.operators.fidelity import Fidelity
from fplab.prox.prox_icnn import ICNNProxSolver


@dataclass
class FISTATrace:
    residuals: list[float]
    objectives: list[float]
    prox_grad_norms: list[float]
    prox_thresholds: list[float]
    momenta: list[float]
    restarts: list[bool]


class FISTAProxGradSolver:
    """Accelerated proximal-gradient solver (FISTA) with optional monotone safeguard."""

    def __init__(self, fidelity: Fidelity, regularizer: ICNNRegularizer, prox_solver: ICNNProxSolver) -> None:
        self.fidelity = fidelity
        self.regularizer = regularizer
        self.prox_solver = prox_solver

    def objective(self, x: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
        return self.fidelity.value(x, y) + lam * self.regularizer(x)

    def _alpha(self) -> float:
        Lf = self.fidelity.lipschitz()
        if Lf <= 0:
            raise ValueError("fidelity lipschitz constant must be positive")
        return 1.0 / Lf

    def _prox_step(
        self,
        x_ref: torch.Tensor,
        y: torch.Tensor,
        lam: float,
        alpha: float,
        differentiable: bool,
    ) -> tuple[torch.Tensor, float, float]:
        v = x_ref - alpha * self.fidelity.grad(x_ref, y)
        x_next, info = self.prox_solver.prox(
            v,
            alpha=alpha,
            lam=lam,
            regularizer=self.regularizer,
            differentiable=differentiable,
        )
        return x_next, info.grad_norm, info.threshold

    def solve(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        lam: float,
        max_iter: int = 100,
        tol: float = 1e-5,
        differentiable: bool = False,
        early_stop: bool = True,
        monotone: bool = True,
        adaptive_restart: bool = True,
    ) -> tuple[torch.Tensor, FISTATrace]:
        if lam < 0:
            raise ValueError("lam must be nonnegative")

        alpha = self._alpha()
        x_prev = x0.clone()
        x = x0.clone()
        t = 1.0

        residuals: list[float] = []
        objectives: list[float] = []
        prox_grad_norms: list[float] = []
        prox_thresholds: list[float] = []
        momenta: list[float] = []
        restarts: list[bool] = []

        obj_x = torch.mean(self.objective(x, y, lam))

        for _ in range(max_iter):
            restarted = False
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_next
            y_k = x + beta * (x - x_prev)

            x_next, prox_grad_norm, prox_threshold = self._prox_step(
                x_ref=y_k,
                y=y,
                lam=lam,
                alpha=alpha,
                differentiable=differentiable,
            )

            # Adaptive restart is done in non-differentiable mode to avoid
            # graph-discontinuous control flow during unrolled training.
            if adaptive_restart and not differentiable:
                s_k = x_next - x
                d_k = x - x_prev
                if torch.sum(s_k * d_k).item() > 0:
                    restarted = True
                    t_next = 1.0
                    beta = 0.0
                    y_k = x
                    x_next, prox_grad_norm, prox_threshold = self._prox_step(
                        x_ref=y_k,
                        y=y,
                        lam=lam,
                        alpha=alpha,
                        differentiable=differentiable,
                    )

            obj_next = torch.mean(self.objective(x_next, y, lam))

            # Monotone safeguard: if acceleration increases objective, fallback to a
            # plain proximal-gradient step. If that still fails, keep current iterate.
            if monotone and not differentiable and obj_next.item() > obj_x.item() + 1e-8:
                restarted = True
                x_pg, pg_grad_norm, pg_threshold = self._prox_step(
                    x_ref=x,
                    y=y,
                    lam=lam,
                    alpha=alpha,
                    differentiable=differentiable,
                )
                obj_pg = torch.mean(self.objective(x_pg, y, lam))

                if obj_pg.item() <= obj_next.item():
                    x_next = x_pg
                    prox_grad_norm = pg_grad_norm
                    prox_threshold = pg_threshold
                    obj_next = obj_pg

                if obj_next.item() > obj_x.item() + 1e-8:
                    x_next = x
                    prox_grad_norm = 0.0
                    prox_threshold = 0.0
                    obj_next = obj_x

                t_next = 1.0
                beta = 0.0

            residual = float(torch.linalg.norm(x_next - x).item())
            residuals.append(residual)
            prox_grad_norms.append(prox_grad_norm)
            prox_thresholds.append(prox_threshold)
            objectives.append(float(obj_next.item()))
            momenta.append(float(beta))
            restarts.append(restarted)

            x_prev, x = x, x_next
            obj_x = obj_next
            t = t_next

            if early_stop and residual <= tol and prox_grad_norm <= max(tol, 1e-6):
                break

        return x, FISTATrace(
            residuals=residuals,
            objectives=objectives,
            prox_grad_norms=prox_grad_norms,
            prox_thresholds=prox_thresholds,
            momenta=momenta,
            restarts=restarts,
        )
