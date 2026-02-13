# uav_pde_sim (PyTorch)

Coupled PDE–ODE UAV simulation in PyTorch.

What this repo contains:
- 2D wind field evolving via advection–diffusion PDE on a grid
- 2D point-mass UAV dynamics (ODE) coupled to the wind field
- PID controller for trajectory tracking
- Everything runs in torch (CPU/GPU). Wind sampling uses torch.nn.functional.grid_sample.

Install:
  pip install -r requirements.txt

Run the demo:
  PYTHONPATH=src python experiments/exp_01_pid_vs_wind.py

Outputs saved to assets/:
- trajectory.png
- tracking_error.png
- wind_mag_t*.png

Notes:
- Wind sampling uses a pure-torch bilinear interpolator (periodic wrap) instead of torch.nn.functional.grid_sample.
  Reason: some macOS CPU PyTorch builds can segfault in grid_sample; this avoids that native kernel path.
