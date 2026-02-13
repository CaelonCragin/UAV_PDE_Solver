from __future__ import annotations
from dataclasses import dataclass
import torch
from torch_uav.numerics.fd_operators import laplacian_2d, upwind_grad_2d

@dataclass
class WindPDEConfig:
    H: int = 128
    W: int = 128
    Lx: float = 10.0
    Ly: float = 10.0
    nu: float = 0.08
    advect_vel: tuple[float, float] = (0.6, 0.2)
    source_strength: float = 1.5
    source_sigma: float = 0.35
    source_period: float = 2.5

class WindFieldPDE:
    """
    Wind field w(x,y,t) with components (wx, wy):
      w_t + u·∇w = nu ∇²w + S(x,y,t)
    Periodic boundaries (torch.roll stencils).
    """

    def __init__(self, cfg: WindPDEConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.dx = cfg.Lx / cfg.W
        self.dy = cfg.Ly / cfg.H

        # wind: (2, H, W)
        self.wind = torch.zeros((2, cfg.H, cfg.W), device=device, dtype=torch.float32)

        xs = torch.linspace(0.0, cfg.Lx, cfg.W, device=device)
        ys = torch.linspace(0.0, cfg.Ly, cfg.H, device=device)
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        self.X = X
        self.Y = Y

        self._last_source_time = -1e9

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            torch.manual_seed(seed)
        self.wind.zero_()
        self._last_source_time = -1e9

    def _gust_source(self, t: float) -> torch.Tensor:
        cfg = self.cfg
        S = torch.zeros_like(self.wind)

        if (t - self._last_source_time) < cfg.source_period:
            return S

        self._last_source_time = t

        cx = (torch.rand((), device=self.device) * cfg.Lx).item()
        cy = (torch.rand((), device=self.device) * cfg.Ly).item()

        sigma2 = cfg.source_sigma * cfg.source_sigma
        G = torch.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2.0 * sigma2))

        theta = (2.0 * torch.pi * torch.rand((), device=self.device)).item()
        dir_vec = torch.tensor([
            torch.cos(torch.tensor(theta, device=self.device)),
            torch.sin(torch.tensor(theta, device=self.device))
        ], device=self.device)

        S[0] = cfg.source_strength * dir_vec[0] * G
        S[1] = cfg.source_strength * dir_vec[1] * G
        return S

    def rhs(self, wind: torch.Tensor, t: float) -> torch.Tensor:
        cfg = self.cfg
        ux, uy = cfg.advect_vel
        vel_x = torch.tensor(ux, device=self.device, dtype=wind.dtype)
        vel_y = torch.tensor(uy, device=self.device, dtype=wind.dtype)

        dwdx0, dwdy0 = upwind_grad_2d(wind[0], vel_x, vel_y, self.dx, self.dy)
        dwdx1, dwdy1 = upwind_grad_2d(wind[1], vel_x, vel_y, self.dx, self.dy)
        adv0 = vel_x * dwdx0 + vel_y * dwdy0
        adv1 = vel_x * dwdx1 + vel_y * dwdy1

        lap0 = laplacian_2d(wind[0], self.dx, self.dy)
        lap1 = laplacian_2d(wind[1], self.dx, self.dy)

        S = self._gust_source(t)

        rhs = torch.zeros_like(wind)
        rhs[0] = -adv0 + cfg.nu * lap0 + S[0]
        rhs[1] = -adv1 + cfg.nu * lap1 + S[1]
        return rhs

    def step_euler(self, t: float, dt: float) -> None:
        self.wind = self.wind + dt * self.rhs(self.wind, t)

    @torch.no_grad()
    def max_cfl(self, dt: float) -> float:
        cfg = self.cfg
        ux, uy = cfg.advect_vel
        adv_cfl = max(abs(ux) * dt / self.dx, abs(uy) * dt / self.dy)
        diff_cfl = cfg.nu * dt * (2.0/(self.dx*self.dx) + 2.0/(self.dy*self.dy))
        return float(max(adv_cfl, diff_cfl))
