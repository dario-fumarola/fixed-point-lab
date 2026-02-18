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


class ICNNProxSolver:
    """Compute prox_{alpha*lam*R}(v) using differentiable gradient descent."""

    def __init__(self, config: ProxConfig | None = None) -> None:
        self.config = config or ProxConfig()

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

        for it in range(1, self.config.max_iters + 1):
            obj = 0.5 * torch.sum((x - v) ** 2, dim=-1) + alpha * lam * regularizer(x)
            total = torch.mean(obj)
            (grad,) = torch.autograd.grad(total, x, create_graph=differentiable)
            x = x - self.config.lr * grad
            if not differentiable:
                x = x.detach().requires_grad_(True)
            last_grad_norm = float(torch.linalg.norm(grad).item())

            if last_grad_norm <= self.config.tol:
                converged = True
                break

        result = x if differentiable else x.detach()
        return result, ProxStopInfo(iters=it, grad_norm=last_grad_norm, converged=converged)
