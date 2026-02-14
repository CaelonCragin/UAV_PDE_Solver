from __future__ import annotations
from dataclasses import dataclass
import torch

def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-12)

def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # Hamilton product, q and r are (...,4) with [w,x,y,z]
    qw, qx, qy, qz = q.unbind(-1)
    rw, rx, ry, rz = r.unbind(-1)
    return torch.stack([
        qw*rw - qx*rx - qy*ry - qz*rz,
        qw*rx + qx*rw + qy*rz - qz*ry,
        qw*ry - qx*rz + qy*rw + qz*rx,
        qw*rz + qx*ry - qy*rx + qz*rw,
    ], dim=-1)

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    # q (...,4) [w,x,y,z] -> R (...,3,3)
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)

    ww = w*w; xx = x*x; yy = y*y; zz = z*z
    wx = w*x; wy = w*y; wz = w*z
    xy = x*y; xz = x*z; yz = y*z

    R = torch.stack([
        torch.stack([ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy)], dim=-1),
        torch.stack([2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx)], dim=-1),
        torch.stack([2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz], dim=-1),
    ], dim=-2)
    return R

@dataclass
class QuadrotorConfig:
    m: float = 1.2
    Ixx: float = 0.02
    Iyy: float = 0.02
    Izz: float = 0.04
    g: float = 9.81
    thrust_min: float = 0.0
    thrust_max: float = 25.0
    tau_max: float = 1.0

class Quadrotor:
    """
    State x = [p(3), v(3), q(4), w(3)]  -> total 13
    Controls u = [T, tau_x, tau_y, tau_z]
    World frame: z up
    Body frame: thrust along +body z axis.
    """

    def __init__(self, cfg: QuadrotorConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.I = torch.diag(torch.tensor([cfg.Ixx, cfg.Iyy, cfg.Izz], device=device, dtype=torch.float32))
        self.Iinv = torch.diag(torch.tensor([1.0/cfg.Ixx, 1.0/cfg.Iyy, 1.0/cfg.Izz], device=device, dtype=torch.float32))

    def clamp_u(self, u: torch.Tensor) -> torch.Tensor:
        T = u[..., 0].clamp(self.cfg.thrust_min, self.cfg.thrust_max)
        tau = u[..., 1:4].clamp(-self.cfg.tau_max, self.cfg.tau_max)
        return torch.cat([T.unsqueeze(-1), tau], dim=-1)

    def rhs(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        u = self.clamp_u(u)

        p = x[..., 0:3]
        v = x[..., 3:6]
        q = x[..., 6:10]
        w = x[..., 10:13]

        q = quat_normalize(q)
        R = quat_to_rotmat(q)  # (...,3,3)

        # Thrust in body frame -> world
        T = u[..., 0]
        thrust_b = torch.stack([torch.zeros_like(T), torch.zeros_like(T), T], dim=-1)  # (...,3)
        thrust_w = (R @ thrust_b.unsqueeze(-1)).squeeze(-1)  # (...,3)

        # Translational dynamics
        p_dot = v
        v_dot = thrust_w / cfg.m + torch.tensor([0.0, 0.0, -cfg.g], device=self.device, dtype=x.dtype)

        # Quaternion kinematics: q_dot = 0.5 * q ⊗ [0, w]
        w_quat = torch.cat([torch.zeros_like(w[..., :1]), w], dim=-1)
        q_dot = 0.5 * quat_mul(q, w_quat)

        # Rotational dynamics: w_dot = I^{-1}(tau - w × Iw)
        tau = u[..., 1:4]
        Iw = (self.I @ w.unsqueeze(-1)).squeeze(-1)
        w_cross_Iw = torch.cross(w, Iw, dim=-1)
        w_dot = (self.Iinv @ (tau - w_cross_Iw).unsqueeze(-1)).squeeze(-1)

        return torch.cat([p_dot, v_dot, q_dot, w_dot], dim=-1)
