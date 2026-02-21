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
    step_sizes: list[float]
    backtracks: list[int]
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

    def _line_search_step(
        self,
        x_ref: torch.Tensor,
        y: torch.Tensor,
        lam: float,
        alpha0: float,
        differentiable: bool,
        backtrack_factor: float = 0.5,
        max_backtracks: int = 12,
    ) -> tuple[torch.Tensor, float, float, float, int]:
        if not (0.0 < backtrack_factor < 1.0):
            raise ValueError("backtrack_factor must be in (0, 1)")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be nonnegative")

        f_x = self.fidelity.value(x_ref, y)
        grad = self.fidelity.grad(x_ref, y)

        chosen_x = x_ref
        chosen_grad_norm = float("inf")
        chosen_threshold = float("inf")
        chosen_alpha = alpha0
        chosen_backtracks = max_backtracks

        accepted = False

        for backtracks in range(max_backtracks + 1):
            alpha = alpha0 * (backtrack_factor**backtracks)
            x_next, info_grad_norm, info_threshold = self._prox_step(
                x_ref=x_ref,
                y=y,
                lam=lam,
                alpha=alpha,
                differentiable=differentiable,
            )

            diff = x_next - x_ref
            majorizer = f_x + torch.sum(grad * diff, dim=-1) + (0.5 / alpha) * torch.sum(diff * diff, dim=-1)
            lhs = torch.mean(self.fidelity.value(x_next, y))
            rhs = torch.mean(majorizer)

            if lhs.item() <= rhs.item() + 1e-8:
                accepted = True
                chosen_x = x_next
                chosen_grad_norm = info_grad_norm
                chosen_threshold = info_threshold
                chosen_alpha = alpha
                chosen_backtracks = backtracks
                break

            chosen_x = x_next
            chosen_grad_norm = info_grad_norm
            chosen_threshold = info_threshold
            chosen_alpha = alpha
            chosen_backtracks = backtracks

        if not accepted:
            # Keep the smallest attempted step if Armijo-style condition did not pass.
            chosen_x, chosen_grad_norm, chosen_threshold = self._prox_step(
                x_ref=x_ref,
                y=y,
                lam=lam,
                alpha=chosen_alpha,
                differentiable=differentiable,
            )
            chosen_backtracks = max_backtracks

        return chosen_x, chosen_grad_norm, chosen_threshold, chosen_alpha, chosen_backtracks

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
        line_search: bool = False,
        alpha_scale: float = 1.0,
        backtrack_factor: float = 0.5,
        max_backtracks: int = 12,
    ) -> tuple[torch.Tensor, FISTATrace]:
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        if line_search and differentiable:
            raise ValueError("line_search is only supported with differentiable=False")
        if alpha_scale <= 0:
            raise ValueError("alpha_scale must be positive")
        if not (0.0 < backtrack_factor < 1.0):
            raise ValueError("backtrack_factor must be in (0, 1)")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be nonnegative")

        alpha = alpha_scale * self._alpha()
        x_prev = x0.clone()
        x = x0.clone()
        t = 1.0

        residuals: list[float] = []
        objectives: list[float] = []
        prox_grad_norms: list[float] = []
        prox_thresholds: list[float] = []
        step_sizes: list[float] = []
        backtracks: list[int] = []
        momenta: list[float] = []
        restarts: list[bool] = []

        obj_x = torch.mean(self.objective(x, y, lam))

        for _ in range(max_iter):
            restarted = False
            t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
            beta = (t - 1.0) / t_next
            y_k = x + beta * (x - x_prev)

            if line_search:
                x_next, prox_grad_norm, prox_threshold, step_size, n_backtracks = self._line_search_step(
                    x_ref=y_k,
                    y=y,
                    lam=lam,
                    alpha0=alpha,
                    differentiable=differentiable,
                    backtrack_factor=backtrack_factor,
                    max_backtracks=max_backtracks,
                )
            else:
                x_next, prox_grad_norm, prox_threshold = self._prox_step(
                    x_ref=y_k,
                    y=y,
                    lam=lam,
                    alpha=alpha,
                    differentiable=differentiable,
                )
                step_size = alpha
                n_backtracks = 0

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
                    if line_search:
                        x_next, prox_grad_norm, prox_threshold, step_size, n_backtracks = self._line_search_step(
                            x_ref=y_k,
                            y=y,
                            lam=lam,
                            alpha0=alpha,
                            differentiable=differentiable,
                            backtrack_factor=backtrack_factor,
                            max_backtracks=max_backtracks,
                        )
                    else:
                        x_next, prox_grad_norm, prox_threshold = self._prox_step(
                            x_ref=y_k,
                            y=y,
                            lam=lam,
                            alpha=alpha,
                            differentiable=differentiable,
                        )
                        step_size = alpha
                        n_backtracks = 0

            obj_next = torch.mean(self.objective(x_next, y, lam))

            # Monotone safeguard: if acceleration increases objective, fallback to a
            # plain proximal-gradient step. If that still fails, keep current iterate.
            if monotone and not differentiable and obj_next.item() > obj_x.item() + 1e-8:
                restarted = True
                if line_search:
                    x_pg, pg_grad_norm, pg_threshold, step_size, n_backtracks = self._line_search_step(
                        x_ref=x,
                        y=y,
                        lam=lam,
                        alpha0=alpha,
                        differentiable=differentiable,
                        backtrack_factor=backtrack_factor,
                        max_backtracks=max_backtracks,
                    )
                else:
                    x_pg, pg_grad_norm, pg_threshold = self._prox_step(
                        x_ref=x,
                        y=y,
                        lam=lam,
                        alpha=alpha,
                        differentiable=differentiable,
                    )
                    step_size = alpha
                    n_backtracks = 0
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
                    step_size = 0.0
                    n_backtracks = 0
                    obj_next = obj_x

                t_next = 1.0
                beta = 0.0

            residual = float(torch.linalg.norm(x_next - x).item())
            residuals.append(residual)
            prox_grad_norms.append(prox_grad_norm)
            prox_thresholds.append(prox_threshold)
            objectives.append(float(obj_next.item()))
            step_sizes.append(float(step_size))
            backtracks.append(int(n_backtracks))
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
            step_sizes=step_sizes,
            backtracks=backtracks,
            momenta=momenta,
            restarts=restarts,
        )
