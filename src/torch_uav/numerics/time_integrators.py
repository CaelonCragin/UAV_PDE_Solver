from __future__ import annotations
from typing import Callable
import torch

RHS = Callable[[torch.Tensor, float], torch.Tensor]

def euler_step(x: torch.Tensor, t: float, dt: float, f: RHS) -> torch.Tensor:
    """Forward Euler: x_{n+1} = x_n + dt f(x_n, t_n)."""
    return x + dt * f(x, t)

def rk4_step(x: torch.Tensor, t: float, dt: float, f: RHS) -> torch.Tensor:
    """Classic RK4."""
    k1 = f(x, t)
    k2 = f(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
