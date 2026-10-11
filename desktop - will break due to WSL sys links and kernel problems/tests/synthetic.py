"""Tiny analytic ray-caster: renders the depth image + mask of an upright cylinder standing on a
table plane (z = 0), as seen by a pinhole camera. Gives perception tests a ground truth that is
independent of Isaac Sim, so they run anywhere in milliseconds."""

from __future__ import annotations

import numpy as np

from pnp_core.geometry import look_at_optical, make_transform
from pnp_core.perception_math import Intrinsics


def render_cylinder(
    K: Intrinsics,
    width: int,
    height: int,
    eye,
    target,
    center_xy,
    radius: float,
    cyl_height: float,
):
    """Returns (depth_m [H,W] float32 along optical z, mask [H,W] bool, T_base_cam 4x4)."""
    R = look_at_optical(eye, target)
    T_base_cam = make_transform(R, eye)
    us, vs = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    dirs_cam = np.stack([(us - K.cx) / K.fx, (vs - K.cy) / K.fy, np.ones_like(us)], axis=-1)  # z = 1
    d = dirs_cam @ R.T                                   # ray direction in base (scaled so cam-z = 1)
    o = np.asarray(eye, float)
    cx, cy = center_xy

    t_best = np.full(us.shape, np.inf)
    hit_obj = np.zeros(us.shape, bool)

    # table plane z = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t_tab = -o[2] / d[..., 2]
    t_tab[(t_tab <= 0) | ~np.isfinite(t_tab)] = np.inf
    t_best = np.minimum(t_best, t_tab)

    # cylinder side: (ox + t dx - cx)^2 + (oy + t dy - cy)^2 = r^2, 0 <= z <= h
    ox, oy = o[0] - cx, o[1] - cy
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (ox * d[..., 0] + oy * d[..., 1])
    c = ox**2 + oy**2 - radius**2
    disc = b**2 - 4 * a * c
    with np.errstate(invalid="ignore", divide="ignore"):
        t_side = (-b - np.sqrt(disc)) / (2 * a)
    z_side = o[2] + t_side * d[..., 2]
    side_ok = (disc >= 0) & (t_side > 0) & (z_side >= 0) & (z_side <= cyl_height)
    t_side = np.where(side_ok, t_side, np.inf)

    # top cap z = h
    with np.errstate(divide="ignore", invalid="ignore"):
        t_top = (cyl_height - o[2]) / d[..., 2]
    px, py = o[0] + t_top * d[..., 0], o[1] + t_top * d[..., 1]
    top_ok = (t_top > 0) & ((px - cx) ** 2 + (py - cy) ** 2 <= radius**2)
    t_top = np.where(top_ok, t_top, np.inf)

    t_obj = np.minimum(t_side, t_top)
    hit_obj = t_obj < t_best
    t_best = np.minimum(t_best, t_obj)

    # dirs_cam has z = 1, so the ray parameter t IS the optical-axis depth
    depth = np.where(np.isfinite(t_best), t_best, 0.0).astype(np.float32)
    return depth, hit_obj, T_base_cam
