from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def save_quad_frame_xy(
    p: np.ndarray, pref: np.ndarray,
    outpath: str, title: str = ""
) -> None:
    plt.figure()
    plt.plot(pref[:, 0], pref[:, 1], "--", linewidth=1)
    plt.scatter([p[0]], [p[1]], s=60)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()
