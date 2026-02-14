from __future__ import annotations
import os
import shutil
import numpy as np
import torch

from torch_uav.numerics.time_integrators import rk4_step
from torch_uav.dynamics.quadrotor import Quadrotor, QuadrotorConfig, quat_normalize
from torch_uav.control.quad_cascaded import CascadedController, CascadedConfig
from torch_uav.viz.animate import make_gif
from torch_uav.viz.quad_animate import save_quad_frame_xy
from torch_uav.viz.quad_plots import ensure_dir, plot_xyz, plot_inputs, plot_xy_path

def ref_traj(t: float) -> tuple[np.ndarray, np.ndarray]:
    # Hover at z=1, step in x after 2s, small y sine after 3s
    z = 1.0
    if t < 2.0:
        x = 0.0
    else:
        x = 0.6
    y = 0.2 * np.sin(0.6 * max(t - 3.0, 0.0))
    # approximate ref velocity
    vx = 0.0
    vy = 0.2 * 0.6 * np.cos(0.6 * max(t - 3.0, 0.0)) if t > 3.0 else 0.0
    vz = 0.0
    return np.array([x, y, z], dtype=np.float32), np.array([vx, vy, vz], dtype=np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    outdir = "assets"
    ensure_dir(outdir)

    frames_dir = os.path.join(outdir, "frames_quad")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    qcfg = QuadrotorConfig()
    quad = Quadrotor(qcfg, device=device)

    ccfg = CascadedConfig(
        kp_pos=4.0, kd_pos=3.0,
        kp_att=8.0, kd_att=0.20,
        yaw_ref=0.0
    )
    ctrl = CascadedController(ccfg, m=qcfg.m, g=qcfg.g, device=device)

    # State: [p(3), v(3), q(4), w(3)]
    p0 = torch.tensor([0.0, 0.0, 0.0], device=device)
    v0 = torch.tensor([0.0, 0.0, 0.0], device=device)
    q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # [w,x,y,z]
    w0 = torch.tensor([0.0, 0.0, 0.0], device=device)
    x = torch.cat([p0, v0, q0, w0], dim=-1)

    Tfinal = 8.0
    dt = 0.002
    steps = int(Tfinal / dt)

    p_hist = np.zeros((steps, 3), dtype=np.float32)
    pref_hist = np.zeros((steps, 3), dtype=np.float32)
    u_hist = np.zeros((steps, 4), dtype=np.float32)
    t_hist = np.zeros((steps,), dtype=np.float32)

    save_every = 10  # frames every 10 steps -> 400 frames over 8s
    frame_idx = 0

    for k in range(steps):
        t = k * dt
        t_hist[k] = t

        p = x[0:3]
        v = x[3:6]
        q = quat_normalize(x[6:10])
        w = x[10:13]

        pref_np, vref_np = ref_traj(t)
        p_ref = torch.tensor(pref_np, device=device)
        v_ref = torch.tensor(vref_np, device=device)

        u = ctrl(p, v, q, w, p_ref, v_ref)

        def f(x_local: torch.Tensor, t_local: float):
            return quad.rhs(x_local, u)

        x = rk4_step(x, t, dt, f)
        x[6:10] = quat_normalize(x[6:10])

        p_hist[k] = x[0:3].detach().cpu().numpy()
        pref_hist[k] = pref_np
        u_hist[k] = u.detach().cpu().numpy()

        if (k % save_every) == 0:
            # Draw XY frame with full ref path so far
            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            save_quad_frame_xy(
                p=p_hist[k],
                pref=pref_hist[:k+1],
                outpath=frame_path,
                title=f"t = {t:.2f}s"
            )
            frame_idx += 1

    plot_xyz(t_hist, p_hist, os.path.join(outdir, "quad_xyz.png"))
    plot_xy_path(p_hist, pref_hist, os.path.join(outdir, "quad_xy.png"))
    plot_inputs(t_hist, u_hist, os.path.join(outdir, "quad_inputs.png"))

    gif_path = os.path.join(outdir, "demo_quad_xy.gif")
    make_gif(frames_dir, gif_path, fps=25)

    print("Saved:")
    print(" - assets/quad_xyz.png")
    print(" - assets/quad_xy.png")
    print(" - assets/quad_inputs.png")
    print(" - assets/demo_quad_xy.gif")

if __name__ == "__main__":
    main()
