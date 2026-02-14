from __future__ import annotations
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

def save_frame(
    wind: np.ndarray,          # (2,H,W)
    p: np.ndarray,             # (2,)
    p_ref: np.ndarray,         # (2,)
    outpath: str,
    title: str = "",
) -> None:
    """
    Save a single PNG frame visualizing wind magnitude + UAV/ref points.
    """
    mag = np.sqrt(wind[0]**2 + wind[1]**2)

    plt.figure()
    plt.imshow(mag, origin="lower", aspect="auto")
    plt.colorbar(label="|wind|")

    plt.scatter([p_ref[0]], [p_ref[1]], marker="x", s=60)
    plt.scatter([p[0]], [p[1]], marker="o", s=40)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()

def make_gif(frames_dir: str, gif_path: str, fps: int = 20) -> None:
    """
    Build a GIF from PNG frames named frame_000000.png, frame_000001.png, ...
    Uses duration (ms per frame) for compatibility with new imageio.
    """
    import imageio.v2 as imageio  # avoid deprecation warnings

    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if not files:
        raise RuntimeError(f"No frames found in {frames_dir}")

    images = []
    for f in files:
        images.append(imageio.imread(os.path.join(frames_dir, f)))

    # duration is milliseconds per frame
    duration_ms = int(1000 / fps)
    imageio.mimsave(gif_path, images, duration=duration_ms)
