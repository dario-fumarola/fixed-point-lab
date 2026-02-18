from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProxStopInfo:
    iters: int
    grad_norm: float
    converged: bool
