from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass
class PIDConfig:
    kp: float = 6.0
    kd: float = 4.0
    ki: float = 0.6
    integral_limit: float = 2.0

class PIDPositionController:
    def __init__(self, cfg: PIDConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.integral = torch.zeros((2,), device=device)

    def reset(self):
        self.integral.zero_()

    def __call__(self, p: torch.Tensor, v: torch.Tensor, p_ref: torch.Tensor, v_ref: torch.Tensor, dt: float) -> torch.Tensor:
        e_p = p_ref - p
        e_v = v_ref - v

        self.integral = self.integral + e_p * dt
        self.integral = torch.clamp(self.integral, -self.cfg.integral_limit, self.cfg.integral_limit)

        return self.cfg.kp * e_p + self.cfg.kd * e_v + self.cfg.ki * self.integral
