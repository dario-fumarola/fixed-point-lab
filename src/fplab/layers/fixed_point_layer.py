from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from fplab.models.icnn import ICNNRegularizer
from fplab.operators.fidelity import Fidelity
from fplab.prox.prox_icnn import ICNNProxSolver
from fplab.solvers.fista import FISTAProxGradSolver, FISTATrace
from fplab.solvers.proxgrad import ProxGradSolver, SolveTrace


SolverTrace = SolveTrace | FISTATrace
SolverOverrides = dict[str, Any]


@dataclass(frozen=True)
class FixedPointLayerConfig:
    """Configuration that controls solver behavior for the fixed-point layer."""

    solver: str = "pg"
    max_iter: int = 6
    tol: float = 1e-5
    differentiable: bool = True
    early_stop: bool = False
    monotone: bool = False
    adaptive_restart: bool = True
    line_search: bool = False
    alpha_scale: float = 1.0
    backtrack_factor: float = 0.5
    max_backtracks: int = 12


class FixedPointLayer(nn.Module):
    """Differentiable fixed-point layer backed by the repository solvers.

    The layer holds a fidelity term, an ICNN regularizer, and a prox solver, then
    unrolls either proximal-gradient or FISTA updates as a forward pass.
    """

    _solver_default_kwargs_by_name: dict[str, set[str]] = {
        "pg": {
            "x0",
            "y",
            "lam",
            "max_iter",
            "tol",
            "differentiable",
            "early_stop",
            "alpha_scale",
            "backtrack_factor",
            "max_backtracks",
            "line_search",
        },
        "fista": {
            "x0",
            "y",
            "lam",
            "max_iter",
            "tol",
            "differentiable",
            "early_stop",
            "monotone",
            "adaptive_restart",
            "line_search",
            "alpha_scale",
            "backtrack_factor",
            "max_backtracks",
        },
    }

    def __init__(
        self,
        fidelity: Fidelity,
        regularizer: ICNNRegularizer,
        prox_solver: ICNNProxSolver,
        config: FixedPointLayerConfig | None = None,
        *,
        lam: float = 0.1,
        solver_kwargs: SolverOverrides | None = None,
    ) -> None:
        super().__init__()
        layer_cfg = config or FixedPointLayerConfig()
        solver_name = layer_cfg.solver.lower()
        if solver_name not in {"pg", "fista"}:
            raise ValueError(f"unsupported solver: {solver_name}")

        self.lam = float(lam)
        self.config = layer_cfg
        self.solver = solver_name
        self.fidelity = fidelity
        self.regularizer = regularizer
        self.prox_solver = prox_solver
        self.solver_kwargs: SolverOverrides = dict(solver_kwargs or {})
        self._verify_solver_kwargs()

        if solver_name == "pg":
            self.solver_impl = ProxGradSolver(
                fidelity=fidelity,
                regularizer=regularizer,
                prox_solver=prox_solver,
            )
        else:
            self.solver_impl = FISTAProxGradSolver(
                fidelity=fidelity,
                regularizer=regularizer,
                prox_solver=prox_solver,
            )

    def _verify_solver_kwargs(self) -> None:
        unknown = set(self.solver_kwargs).difference(
            self._solver_default_kwargs_by_name.get(self.solver, set())
        )
        if unknown:
            raise ValueError(
                f"unsupported solver argument(s) for {self.solver}: {', '.join(sorted(unknown))}"
            )

    def _build_base_kwargs(
        self,
        x0: torch.Tensor,
        y: torch.Tensor,
        lam: float | None = None,
        overrides: SolverOverrides | None = None,
    ) -> SolverOverrides:
        base = {
            "x0": x0,
            "y": y,
            "lam": self.lam if lam is None else float(lam),
            "max_iter": self.config.max_iter,
            "tol": self.config.tol,
            "differentiable": self.config.differentiable,
            "early_stop": self.config.early_stop,
            "alpha_scale": self.config.alpha_scale,
            "backtrack_factor": self.config.backtrack_factor,
            "max_backtracks": self.config.max_backtracks,
        }

        if self.solver == "pg":
            base["line_search"] = self.config.line_search
        else:
            base.update(
                {
                    "monotone": self.config.monotone,
                    "adaptive_restart": self.config.adaptive_restart,
                    "line_search": self.config.line_search,
                }
            )

        # Layer-level defaults can be overridden per-call.
        base.update(self.solver_kwargs)
        if overrides:
            unknown = set(overrides).difference(self._solver_default_kwargs_by_name[self.solver])
            if unknown:
                raise ValueError(
                    f"unsupported override(s) for {self.solver}: {', '.join(sorted(unknown))}"
                )
            base.update(overrides)

        return base

    def forward(
        self,
        y: torch.Tensor,
        lam: float | None = None,
        x0: torch.Tensor | None = None,
        return_trace: bool = False,
        solver_overrides: SolverOverrides | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, SolverTrace]:
        x0 = torch.zeros_like(y) if x0 is None else x0
        kwargs = self._build_base_kwargs(x0=x0, y=y, lam=lam, overrides=solver_overrides)
        x_hat, trace = self.solver_impl.solve(**kwargs)  # type: ignore[misc]

        if return_trace:
            return x_hat, trace
        return x_hat
