from __future__ import annotations
import torch

def _wrap01(x: torch.Tensor) -> torch.Tensor:
    # wrap x into [0,1)
    return x - torch.floor(x)

def sample_wind_bilinear(wind: torch.Tensor, pos_xy: torch.Tensor, Lx: float, Ly: float) -> torch.Tensor:
    """
    Pure-torch bilinear sampling with periodic wrap. Avoids grid_sample (which can segfault on some mac CPU builds).

    wind: (2, H, W)
    pos_xy: (..., 2) with x in [0,Lx], y in [0,Ly] (we'll wrap periodically anyway)
    Returns: (..., 2)
    """
    assert wind.ndim == 3 and wind.shape[0] == 2, "wind must be (2,H,W)"
    H, W = wind.shape[1], wind.shape[2]

    # Normalize positions to grid coordinates
    x = pos_xy[..., 0] / Lx  # in R
    y = pos_xy[..., 1] / Ly

    # periodic wrap to [0,1)
    x = _wrap01(x)
    y = _wrap01(y)

    gx = x * (W - 1)
    gy = y * (H - 1)

    x0 = torch.floor(gx).to(torch.long)
    y0 = torch.floor(gy).to(torch.long)
    x1 = (x0 + 1) % W
    y1 = (y0 + 1) % H

    # fractional part
    tx = (gx - x0.to(gx.dtype)).clamp(0.0, 1.0)
    ty = (gy - y0.to(gy.dtype)).clamp(0.0, 1.0)

    # gather values for each component
    # wind[c, y, x]
    def g(c, yy, xx):
        return wind[c, yy, xx]

    w00x = g(0, y0, x0); w01x = g(0, y0, x1); w10x = g(0, y1, x0); w11x = g(0, y1, x1)
    w00y = g(1, y0, x0); w01y = g(1, y0, x1); w10y = g(1, y1, x0); w11y = g(1, y1, x1)

    # bilinear interpolation
    wx0 = w00x * (1 - tx) + w01x * tx
    wx1 = w10x * (1 - tx) + w11x * tx
    wx  = wx0 * (1 - ty) + wx1 * ty

    wy0 = w00y * (1 - tx) + w01y * tx
    wy1 = w10y * (1 - tx) + w11y * tx
    wy  = wy0 * (1 - ty) + wy1 * ty

    return torch.stack([wx, wy], dim=-1)
