"""Close look WITHOUT a learned model: refine a small box-shaped object lying on the desk (a LEGO brick) from the wrist
camera's DEPTH alone. Used by pnp_perception/wrist_refine_node.py (the /perception/refine_pose service when there is
no FoundationPose mesh for the object, e.g. the SO-101's bricks; docs/learning/17_so101.md "Wrist camera").

Why depth alone works here: at the pre-grasp pose the D405 is ~13 cm from the brick, where its depth is good to well
under a millimetre, and a brick is the only thing standing up from the flat desk near the first estimate. So:
  1. points near the first estimate (base frame, from the wrist depth image + TF)
  2. the local desk height: the low points around it (robust to a few mm of calibration error in z)
  3. points standing > min_height above the desk, but not much higher than the object (the gripper's jaws are in
     view right above it) -> a 2 mm occupancy grid -> connected blobs
  4. the blob nearest the estimate; its footprint (cv2.minAreaRect: the top face covers it, the sides seen at an
     angle project onto its edges) gives the center and the yaw; the height checks it is the right object
  5. checks: footprint and height like the object's (else a neighbour touching it, or only part of it seen) -> None
The yaw keeps the first estimate's tilt and picks the symmetric twin (yaw_period_deg) nearest to it.
Pure numpy + OpenCV; tests/test_wrist_refine.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import make_transform, rot_z, wrap_angle
from .object_grasp import object_yaw, support_height, tilt_deg


@dataclass
class RefineResult:
    T_base_obj: np.ndarray | None          # refined pose, None = not trusted (keep the first estimate)
    message: str
    info: dict = field(default_factory=dict)


def refine_on_desk(points_base: np.ndarray, T_init: np.ndarray, obj_cfg: dict, desk_z: float | None = None,
                   search_radius: float | None = None, min_height: float = 0.003, cell: float = 0.002,
                   size_tol: float = 0.35, desk_tol: float = 0.015) -> RefineResult:
    """points_base: (N, 3) wrist-camera points in the base frame. T_init: the first estimate (object frame, origin at
    its center). desk_z: the configured desk height (a sanity bound for the measured one; None = trust the data)."""
    dims = np.asarray(obj_cfg["dims_m"], float)
    foot = np.sort(dims[:2])[::-1]                                  # (long, short) side of the footprint
    r_obj = 0.5 * float(np.hypot(*foot))
    R0 = search_radius or max(0.03, 3.0 * r_obj)
    c0 = T_init[:3, 3]
    P = np.asarray(points_base, float)
    P = P[np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1]) < R0]
    if len(P) < 200:
        return RefineResult(None, f"only {len(P)} wrist points near the estimate")
    # 2. the desk: the lowest quarter of the points around the estimate, then the points near it
    z_lo = np.percentile(P[:, 2], 25)
    near = P[np.abs(P[:, 2] - z_lo) < 0.004]
    z_desk = float(np.median(near[:, 2]))
    if desk_z is not None and abs(z_desk - desk_z) > desk_tol:
        return RefineResult(None, f"desk at {z_desk * 1000:.1f} mm, expected {desk_z * 1000:.1f} mm")
    # 3. standing points (up to a little above the object's height: the open jaws hang right above it at the
    # pre-grasp pose and would merge with it) -> occupancy grid -> blobs
    upright_h = 2 * support_height(T_init[:3, :3], obj_cfg)
    up = P[(P[:, 2] > z_desk + min_height) & (P[:, 2] < z_desk + upright_h + max(0.006, 0.5 * upright_h))]
    if len(up) < 30:
        return RefineResult(None, "nothing stands up from the desk near the estimate", {"desk_z": z_desk})
    n = int(np.ceil(2 * R0 / cell)) + 1
    ij = np.clip(((up[:, :2] - (c0[:2] - R0)) / cell).astype(int), 0, n - 1)
    grid = np.zeros((n, n), np.uint8)
    grid[ij[:, 1], ij[:, 0]] = 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))     # close 1-cell gaps (sparse rows)
    count, labels = cv2.connectedComponents(grid, connectivity=8)
    lab = labels[ij[:, 1], ij[:, 0]]
    best, best_d = None, np.inf
    for k in range(1, count):
        pts = up[lab == k]
        if len(pts) < 30:
            continue
        d = float(np.hypot(*(np.median(pts[:, :2], axis=0) - c0[:2])))
        if d < best_d:
            best, best_d = pts, d
    if best is None:
        return RefineResult(None, "no blob of the object's size near the estimate", {"desk_z": z_desk})
    # 4. footprint -> center + yaw; height
    (cx, cy), (w, h), ang = cv2.minAreaRect(best[:, :2].astype(np.float32))
    long_, short = max(w, h), min(w, h)
    yaw_long = np.radians(ang if w >= h else ang + 90.0)
    height = float(np.percentile(best[:, 2], 95) - z_desk)
    info = {"desk_z": z_desk, "footprint_m": [round(long_, 4), round(short, 4)], "height_m": round(height, 4),
            "points": int(len(best)), "offset_m": round(best_d, 4)}
    # 5. is it the object? (a neighbour touching it makes the blob too long; a partly hidden one too short)
    if not (abs(long_ - foot[0]) < size_tol * foot[0] + 0.003 and abs(short - foot[1]) < size_tol * foot[1] + 0.003):
        return RefineResult(None, f"blob {long_ * 1000:.0f} x {short * 1000:.0f} mm is not the object's footprint "
                                  f"{foot[0] * 1000:.0f} x {foot[1] * 1000:.0f} mm", info)
    if abs(height - upright_h) > max(0.004, 0.35 * upright_h):
        return RefineResult(None, f"blob height {height * 1000:.1f} mm, object {upright_h * 1000:.1f} mm", info)
    z = z_desk + support_height(T_init[:3, :3], obj_cfg)
    R = T_init[:3, :3]
    if tilt_deg(R, obj_cfg) < 20.0:                       # lying flat as expected: the footprint gives the yaw
        # the object's x axis is its longer side (dims_m) or its shorter one
        yaw_x = yaw_long if dims[0] >= dims[1] else yaw_long + np.pi / 2
        period = np.radians(float(obj_cfg.get("grasp", {}).get("yaw_period_deg", 360.0)))
        if abs(foot[0] - foot[1]) < 0.002:                # square: the footprint's sides fix the yaw mod 90 deg only
            period = min(period, np.pi / 2)
        y0 = object_yaw(R, obj_cfg)
        k = np.round(wrap_angle(y0 - yaw_x) / period) if period < 2 * np.pi - 1e-6 else 0.0
        yaw = yaw_x + k * period
        # the footprint's own ambiguity is 180 deg even for a non-symmetric object: take the twin nearest y0
        if abs(wrap_angle(yaw - y0)) > np.pi / 2:
            yaw += np.pi
        R = rot_z(wrap_angle(yaw - y0)) @ R
        info["yaw_change_deg"] = round(float(np.degrees(wrap_angle(yaw - y0))), 2)
    return RefineResult(make_transform(R, [cx, cy, z]), "ok", info)
