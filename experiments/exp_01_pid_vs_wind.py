from __future__ import annotations
import os
import shutil
import numpy as np
import torch

from torch_uav.pde.wind_field import WindPDEConfig, WindFieldPDE
from torch_uav.dynamics.uav_models import PointMassUAV, PointMassUAVConfig
from torch_uav.control.pid import PIDPositionController, PIDConfig
from torch_uav.coupling.wind_sampling import sample_wind_bilinear
from torch_uav.numerics.time_integrators import rk4_step
from torch_uav.viz.plots import ensure_dir, plot_trajectory, plot_tracking_error, plot_wind_snapshot
from torch_uav.viz.animate import save_frame, make_gif

def reference_circle(t: float, center=(5.0, 5.0), radius=2.5, omega=0.45):
    cx, cy = center
    x = cx + radius * np.cos(omega * t)
    y = cy + radius * np.sin(omega * t)
    vx = -radius * omega * np.sin(omega * t)
    vy =  radius * omega * np.cos(omega * t)
    return np.array([x, y], dtype=np.float32), np.array([vx, vy], dtype=np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    outdir = "assets"
    ensure_dir(outdir)

    frames_dir = os.path.join(outdir, "frames_pid_wind")
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Wind PDE
    wcfg = WindPDEConfig(
        H=128, W=128, Lx=10.0, Ly=10.0, nu=0.08,
        advect_vel=(0.5, 0.2),
        source_strength=1.8, source_sigma=0.35, source_period=2.0
    )
    wind_pde = WindFieldPDE(wcfg, device=device)
    wind_pde.reset(seed=0)

    # UAV
    ucfg = PointMassUAVConfig(m=1.0, drag_k=0.9, u_max=7.0)
    uav = PointMassUAV(ucfg, device=device)

    # Controller
    pcfg = PIDConfig(kp=7.0, kd=4.5, ki=0.7, integral_limit=2.0)
    pid = PIDPositionController(pcfg, device=device)
    pid.reset()

    x = torch.tensor([5.0, 2.5, 0.0, 0.0], device=device)

    T = 25.0
    dt = 0.01
    steps = int(T / dt)

    p_hist = np.zeros((steps, 2), dtype=np.float32)
    pref_hist = np.zeros((steps, 2), dtype=np.float32)
    err_hist = np.zeros((steps,), dtype=np.float32)
    t_hist = np.zeros((steps,), dtype=np.float32)

    snap_times = {5.0, 12.5, 20.0}

    cfl = wind_pde.max_cfl(dt)
    print(f"Wind PDE CFL-ish indicator (heuristic): {cfl:.3f}")

    # For GIF: save every N steps
    save_every = 5  # 25s / 0.01 = 2500 steps -> 500 frames at every 5 (reasonable)
    frame_idx = 0

    for k in range(steps):
        t = k * dt
        t_hist[k] = t

        wind_pde.step_euler(t, dt)

        p_ref_np, v_ref_np = reference_circle(t)
        p_ref = torch.tensor(p_ref_np, device=device)
        v_ref = torch.tensor(v_ref_np, device=device)

        p = x[0:2]
        v = x[2:4]

        u_ctrl = pid(p, v, p_ref, v_ref, dt)

        def f(x_local: torch.Tensor, t_local: float):
            p_local = x_local[0:2]
            w_local = sample_wind_bilinear(wind_pde.wind, p_local.unsqueeze(0), wcfg.Lx, wcfg.Ly).squeeze(0)
            return uav.rhs(x_local, u_ctrl, w_local)

        x = rk4_step(x, t, dt, f)

        # periodic wrap
        x[0] = x[0] % wcfg.Lx
        x[1] = x[1] % wcfg.Ly

        p_hist[k] = x[0:2].detach().cpu().numpy()
        pref_hist[k] = p_ref_np
        err_hist[k] = float(torch.linalg.norm((x[0:2] - p_ref)).detach().cpu())

        # snapshots
        if any(abs(t - s) < 0.5*dt for s in snap_times):
            wind_np = wind_pde.wind.detach().cpu().numpy()
            plot_wind_snapshot(
                wind_np,
                os.path.join(outdir, f"wind_mag_t{t:.1f}.png"),
                title=f"Wind magnitude at t={t:.1f}s"
            )

        # GIF frames
        if (k % save_every) == 0:
            wind_np = wind_pde.wind.detach().cpu().numpy()

            # Map domain coords -> pixel coords for overlay on imshow:
            # x in [0,Lx] -> [0,W-1], y in [0,Ly] -> [0,H-1]
            def to_pix(pxy):
                return np.array([
                    (pxy[0] / wcfg.Lx) * (wcfg.W - 1),
                    (pxy[1] / wcfg.Ly) * (wcfg.H - 1),
                ], dtype=np.float32)

            p_pix = to_pix(p_hist[k])
            pref_pix = to_pix(p_ref_np)

            frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.png")
            save_frame(
                wind_np,
                p=p_pix,
                p_ref=pref_pix,
                outpath=frame_path,
                title=f"t = {t:.2f}s"
            )
            frame_idx += 1

    plot_trajectory(p_hist, pref_hist, os.path.join(outdir, "trajectory.png"))
    plot_tracking_error(err_hist, t_hist, os.path.join(outdir, "tracking_error.png"))

    gif_path = os.path.join(outdir, "demo_pid_wind.gif")
    make_gif(frames_dir, gif_path, fps=20)

    print("Saved outputs to assets/")
    print("GIF:", gif_path)

if __name__ == "__main__":
    main()
