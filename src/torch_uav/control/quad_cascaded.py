from __future__ import annotations
from dataclasses import dataclass
import torch
from torch_uav.dynamics.quadrotor import quat_normalize, quat_to_rotmat

@dataclass
class CascadedConfig:
    # outer loop
    kp_pos: float = 4.0
    kd_pos: float = 3.0

    # inner loop (attitude)
    kp_att: float = 8.0
    kd_att: float = 0.20

    yaw_ref: float = 0.0  # radians

class CascadedController:
    """
    Cascaded position -> attitude controller.

    Outer loop:
      a_cmd = kp (p_ref - p) + kd (v_ref - v) + [0,0,g]
    Inner loop:
      Build R_des from z_b_des aligned with a_cmd and a fixed yaw reference.
      Use geometric attitude error:
        e_R = 0.5 vee(R_des^T R - R^T R_des)
      Torque:
        tau = -kp_att e_R - kd_att w

    Thrust:
      T = m * (a_cmd · z_b_world)
    """

    def __init__(self, cfg: CascadedConfig, m: float, g: float, device: torch.device):
        self.cfg = cfg
        self.m = m
        self.g = g
        self.device = device

    def __call__(
        self,
        p: torch.Tensor, v: torch.Tensor, q: torch.Tensor, w: torch.Tensor,
        p_ref: torch.Tensor, v_ref: torch.Tensor
    ) -> torch.Tensor:
        cfg = self.cfg

        q = quat_normalize(q)
        R = quat_to_rotmat(q)  # (3,3)
        z_b_world = R[:, 2]    # body z axis in world coords

        # Outer loop accel command (world)
        e_p = p_ref - p
        e_v = v_ref - v
        a_cmd = cfg.kp_pos * e_p + cfg.kd_pos * e_v + torch.tensor([0.0, 0.0, self.g], device=self.device, dtype=p.dtype)

        # Desired body z (world)
        z_b_des = a_cmd / torch.linalg.norm(a_cmd).clamp_min(1e-9)

        # Desired yaw axis
        yaw = torch.tensor(cfg.yaw_ref, device=self.device, dtype=p.dtype)
        x_c = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], dim=-1)

        # Build R_des = [x_b_des, y_b_des, z_b_des] columns
        y_b_des = torch.cross(z_b_des, x_c, dim=-1)
        y_b_des = y_b_des / torch.linalg.norm(y_b_des).clamp_min(1e-9)
        x_b_des = torch.cross(y_b_des, z_b_des, dim=-1)
        R_des = torch.stack([x_b_des, y_b_des, z_b_des], dim=-1)

        # Geometric attitude error
        E = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = torch.stack([E[2, 1], E[0, 2], E[1, 0]], dim=-1)

        # Inner loop torque (dampen angular rates)
        tau = -cfg.kp_att * e_R - cfg.kd_att * w

        # Thrust along current body z axis
        T = self.m * torch.dot(a_cmd, z_b_world)

        return torch.cat([T.view(1), tau.view(3)], dim=-1)
