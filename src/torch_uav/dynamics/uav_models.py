from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class PointMassUAVConfig:
    m: float = 1.0
    drag_k: float = 0.8
    u_max: float = 6.0

class PointMassUAV:
    """
    State x = [px, py, vx, vy]
      p_dot = v
      v_dot = (1/m) u_ctrl - k (v - w(p,t))
    """

    def __init__(self, cfg: PointMassUAVConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def clamp_u(self, u: torch.Tensor) -> torch.Tensor:
        mag = torch.linalg.norm(u, dim=-1, keepdim=True).clamp_min(1e-9)
        scale = torch.clamp(self.cfg.u_max / mag, max=1.0)
        return u * scale

    def rhs(self, x: torch.Tensor, u_ctrl: torch.Tensor, w_samp: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        v = x[..., 2:4]
        u = self.clamp_u(u_ctrl)

        p_dot = v
        v_dot = (1.0 / cfg.m) * u - cfg.drag_k * (v - w_samp)
        return torch.cat([p_dot, v_dot], dim=-1)
