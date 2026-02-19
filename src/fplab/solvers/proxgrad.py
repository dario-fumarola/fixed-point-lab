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
    step_sizes: list[float]
    backtracks: list[int]


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
        alpha_scale: float = 1.0,
        line_search: bool = False,
        backtrack_factor: float = 0.5,
        max_backtracks: int = 12,
    ) -> tuple[torch.Tensor, float, float, float, int]:
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        if alpha_scale <= 0:
            raise ValueError("alpha_scale must be positive")
        if line_search and differentiable:
            raise ValueError("line_search is only supported with differentiable=False")
        if not (0.0 < backtrack_factor < 1.0):
            raise ValueError("backtrack_factor must be in (0, 1)")
        if max_backtracks < 0:
            raise ValueError("max_backtracks must be nonnegative")

        Lf = self.fidelity.lipschitz()
        if Lf <= 0:
            raise ValueError("fidelity lipschitz constant must be positive")

        alpha0 = float(alpha_scale / Lf)
        grad = self.fidelity.grad(x, y)

        if not line_search:
            v = x - alpha0 * grad
            x_next, info = self.prox_solver.prox(
                v,
                alpha=alpha0,
                lam=lam,
                regularizer=self.regularizer,
                differentiable=differentiable,
            )
            return x_next, info.grad_norm, info.threshold, alpha0, 0

        f_x = self.fidelity.value(x, y)
        x_next = x
        info_grad_norm = float("inf")
        info_threshold = float("inf")
        alpha_used = alpha0
        accepted = False

        for backtracks in range(max_backtracks + 1):
            alpha = alpha0 * (backtrack_factor**backtracks)
            v = x - alpha * grad
            candidate, info = self.prox_solver.prox(
                v,
                alpha=alpha,
                lam=lam,
                regularizer=self.regularizer,
                differentiable=False,
            )

            diff = candidate - x
            f_candidate = self.fidelity.value(candidate, y)
            majorizer = f_x + torch.sum(grad * diff, dim=-1) + (0.5 / alpha) * torch.sum(diff * diff, dim=-1)

            lhs = torch.mean(f_candidate)
            rhs = torch.mean(majorizer)
            if lhs.item() <= rhs.item() + 1e-8:
                x_next = candidate
                info_grad_norm = info.grad_norm
                info_threshold = info.threshold
                alpha_used = alpha
                accepted = True
                break

        if not accepted:
            # Use the smallest tried step if Armijo-style condition did not pass.
            alpha_used = alpha0 * (backtrack_factor**max_backtracks)
            v = x - alpha_used * grad
            x_next, info = self.prox_solver.prox(
                v,
                alpha=alpha_used,
                lam=lam,
                regularizer=self.regularizer,
                differentiable=False,
            )
            info_grad_norm = info.grad_norm
            info_threshold = info.threshold
            backtracks = max_backtracks

        return x_next, info_grad_norm, info_threshold, alpha_used, backtracks

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
        alpha_scale: float = 1.0,
        line_search: bool = False,
        backtrack_factor: float = 0.5,
        max_backtracks: int = 12,
    ) -> tuple[torch.Tensor, SolveTrace]:
        x = x0.clone()
        residuals: list[float] = []
        objectives: list[float] = []
        prox_grad_norms: list[float] = []
        prox_thresholds: list[float] = []
        step_sizes: list[float] = []
        backtracks_used: list[int] = []

        for _ in range(max_iter):
            x_next, prox_grad_norm, prox_threshold, alpha, n_backtracks = self.step(
                x,
                y,
                lam,
                differentiable=differentiable,
                alpha_scale=alpha_scale,
                line_search=line_search,
                backtrack_factor=backtrack_factor,
                max_backtracks=max_backtracks,
            )
            residual = float(torch.linalg.norm(x_next - x).item())
            residuals.append(residual)
            prox_grad_norms.append(prox_grad_norm)
            prox_thresholds.append(prox_threshold)
            step_sizes.append(alpha)
            backtracks_used.append(n_backtracks)
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
            step_sizes=step_sizes,
            backtracks=backtracks_used,
        )
