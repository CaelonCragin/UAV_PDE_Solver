from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def plot_trajectory(p_hist: np.ndarray, pref_hist: np.ndarray, outpath: str) -> None:
    plt.figure()
    plt.plot(p_hist[:, 0], p_hist[:, 1], label="UAV")
    plt.plot(pref_hist[:, 0], pref_hist[:, 1], "--", label="Reference")
    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Trajectory")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_tracking_error(err_hist: np.ndarray, t: np.ndarray, outpath: str) -> None:
    plt.figure()
    plt.plot(t, err_hist)
    plt.xlabel("time [s]")
    plt.ylabel("||p - p_ref||")
    plt.title("Tracking Error")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_wind_snapshot(wind: np.ndarray, outpath: str, title: str = "Wind magnitude") -> None:
    mag = np.sqrt(wind[0]**2 + wind[1]**2)
    plt.figure()
    plt.imshow(mag, origin="lower", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
