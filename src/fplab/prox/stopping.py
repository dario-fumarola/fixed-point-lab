from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProxStopInfo:
    """Stopping diagnostics for the inexact prox solve.

    `grad_norm` and `threshold` report the worst-case per-sample values.
    """

    iters: int
    grad_norm: float
    threshold: float
    converged: bool
