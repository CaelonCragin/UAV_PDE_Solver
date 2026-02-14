from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_xyz(t: np.ndarray, p: np.ndarray, outpath: str) -> None:
    plt.figure()
    plt.plot(t, p[:, 0], label="x")
    plt.plot(t, p[:, 1], label="y")
    plt.plot(t, p[:, 2], label="z")
    plt.xlabel("time [s]")
    plt.ylabel("position [m]")
    plt.title("Quad position vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_inputs(t: np.ndarray, u: np.ndarray, outpath: str) -> None:
    plt.figure()
    plt.plot(t, u[:, 0], label="T")
    plt.plot(t, u[:, 1], label="tau_x")
    plt.plot(t, u[:, 2], label="tau_y")
    plt.plot(t, u[:, 3], label="tau_z")
    plt.xlabel("time [s]")
    plt.ylabel("control")
    plt.title("Controls vs time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_xy_path(p: np.ndarray, pref: np.ndarray, outpath: str) -> None:
    plt.figure()
    plt.plot(p[:, 0], p[:, 1], label="quad")
    plt.plot(pref[:, 0], pref[:, 1], "--", label="ref")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("XY path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
