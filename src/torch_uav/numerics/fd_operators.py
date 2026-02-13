from __future__ import annotations
import torch

def _roll(x: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    # x shape: (..., H, W)
    return torch.roll(torch.roll(x, shifts=shift_y, dims=-2), shifts=shift_x, dims=-1)

def laplacian_2d(field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    5-point Laplacian with periodic boundaries via roll.
    field: (..., H, W)
    """
    f = field
    f_xp = _roll(f, 0, -1)
    f_xm = _roll(f, 0, 1)
    f_yp = _roll(f, -1, 0)
    f_ym = _roll(f, 1, 0)
    return (f_xp - 2*f + f_xm) / (dx*dx) + (f_yp - 2*f + f_ym) / (dy*dy)

def upwind_grad_2d(field: torch.Tensor, vel_x: torch.Tensor, vel_y: torch.Tensor, dx: float, dy: float):
    """
    First-order upwind gradient for advection term vel·grad(field).
    field: (..., H, W)
    vel_x, vel_y: broadcastable to field
    Returns: dfdx, dfdy
    """
    f = field
    f_xp = _roll(f, 0, -1)
    f_xm = _roll(f, 0, 1)
    f_yp = _roll(f, -1, 0)
    f_ym = _roll(f, 1, 0)

    dfdx_back = (f - f_xm) / dx
    dfdx_forw = (f_xp - f) / dx
    dfdy_back = (f - f_ym) / dy
    dfdy_forw = (f_yp - f) / dy

    dfdx = torch.where(vel_x >= 0, dfdx_back, dfdx_forw)
    dfdy = torch.where(vel_y >= 0, dfdy_back, dfdy_forw)
    return dfdx, dfdy
