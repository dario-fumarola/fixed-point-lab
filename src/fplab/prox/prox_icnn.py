from __future__ import annotations

from dataclasses import dataclass

import torch

from fplab.models.icnn import ICNNRegularizer
from fplab.prox.stopping import ProxStopInfo


@dataclass
class ProxConfig:
    max_iters: int = 250
    lr: float = 5e-2
    tol: float = 1e-6
    rel_tol: float = 0.0


class ICNNProxSolver:
    """Compute prox_{alpha*lam*R}(v) using differentiable gradient descent."""

    def __init__(self, config: ProxConfig | None = None) -> None:
        self.config = config or ProxConfig()
        if self.config.max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if self.config.lr <= 0:
            raise ValueError("lr must be positive")
        if self.config.tol < 0:
            raise ValueError("tol must be nonnegative")
        if self.config.rel_tol < 0:
            raise ValueError("rel_tol must be nonnegative")

    def prox(
        self,
        v: torch.Tensor,
        alpha: float,
        lam: float,
        regularizer: ICNNRegularizer,
        differentiable: bool = False,
    ) -> tuple[torch.Tensor, ProxStopInfo]:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if lam < 0:
            raise ValueError("lam must be nonnegative")

        init = v if differentiable else v.detach()
        x = init.clone().requires_grad_(True)
        converged = False
        last_grad_norm = float("inf")
        threshold = float("inf")
        iters_ran = 0
        rel_tol = self.config.rel_tol
        abs_tol = self.config.tol

        for it in range(1, self.config.max_iters + 1):
            obj = 0.5 * torch.sum((x - v) ** 2, dim=-1) + alpha * lam * regularizer(x)
            # Sum (not mean) keeps each sample's prox update invariant to batch size.
            total = torch.sum(obj)
            (grad,) = torch.autograd.grad(total, x, create_graph=differentiable)
            x = x - self.config.lr * grad
            if not differentiable:
                x = x.detach().requires_grad_(True)

            grad_norm_per_sample = torch.linalg.vector_norm(grad.detach(), dim=-1)
            if rel_tol > 0.0:
                x_norm_per_sample = torch.linalg.vector_norm(x.detach(), dim=-1)
                threshold_per_sample = torch.clamp(rel_tol * x_norm_per_sample, min=abs_tol)
            else:
                threshold_per_sample = torch.full_like(grad_norm_per_sample, fill_value=abs_tol)

            last_grad_norm = float(torch.max(grad_norm_per_sample).item())
            threshold = float(torch.max(threshold_per_sample).item())
            iters_ran = it

            if bool(torch.all(grad_norm_per_sample <= threshold_per_sample)):
                converged = True
                break

        result = x if differentiable else x.detach()
        return result, ProxStopInfo(
            iters=iters_ran,
            grad_norm=last_grad_norm,
            threshold=threshold,
            converged=converged,
        )
