"""Where should the wrist camera look from to map the desk? (docs/learning/21_wrist_map.md)

The arm samples joint configurations (a 5-joint arm can't take arbitrary camera poses, so we search in joint space
like the wrist calibration, pnp_core/calibration.py), keeps those whose camera looks DOWN at the desk from a good
distance for the D405 (its depth is best within ~0.1-0.4 m), computes what part of the desk each one sees (the image
rectangle -- minus the strip the jaws cover -- projected onto the table), and picks the fewest that cover the work
area (greedy set cover). Pure numpy / OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np


def table_footprint(T_base_opt: np.ndarray, K, size, table_z: float, keep=(0.0, 0.0, 1.0, 1.0),
                    max_range: float | None = None):
    """The part of the table the camera sees: the image rectangle `keep` (fractions u0, v0, u1, v1) projected onto
    the plane z = table_z -> (4, 2) polygon in the base frame, or None (a corner ray misses the table / is farther
    than max_range)."""
    w, h = size
    Ki = np.linalg.inv(np.asarray(K, float))
    out = []
    for u, v in ((keep[0] * w, keep[1] * h), (keep[2] * w, keep[1] * h), (keep[2] * w, keep[3] * h),
                 (keep[0] * w, keep[3] * h)):
        ray = T_base_opt[:3, :3] @ (Ki @ [u, v, 1.0])
        if ray[2] > -1e-6:
            return None
        t = (table_z - T_base_opt[2, 3]) / ray[2]
        p = T_base_opt[:3, 3] + t * ray
        if max_range is not None and np.linalg.norm(p - T_base_opt[:3, 3]) > max_range:
            return None
        out.append(p[:2])
    return np.array(out)


def choose_views(fk_cam, lower, upper, region, K, size, table_z: float, rng, n_samples: int = 20000,
                 dist=(0.12, 0.32), max_view_deg: float = 40.0, min_cam_height: float = 0.10, keep=(0.0, 0.0, 1.0, 0.67),
                 max_range: float = 0.45, max_views: int = 8, cover: float = 0.97, res: float = 0.005, fk_tcp=None,
                 min_tcp_height: float = 0.06):
    """Joint configurations for a desk scan. fk_cam(q) -> T_base_optical. region = (x0, x1, y0, y1).
    -> (list of q in sweep order, coverage fraction of the region)."""
    x0, x1, y0, y1 = region
    nx, ny = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    cands = []
    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        T = fk_cam(q)
        p, z = T[:3, 3], T[:3, 2]
        if p[2] - table_z < min_cam_height or -z[2] < np.cos(np.radians(max_view_deg)):
            continue
        d = (table_z - p[2]) / z[2]                                     # along the optical axis to the table
        if not dist[0] <= d <= dist[1]:
            continue
        if fk_tcp is not None and fk_tcp(q)[2, 3] - table_z < min_tcp_height:
            continue
        poly = table_footprint(T, K, size, table_z, keep, max_range)
        if poly is None:
            continue
        mask = np.zeros((ny, nx), np.uint8)
        cv2.fillPoly(mask, [np.round(np.column_stack([(poly[:, 0] - x0) / res, (poly[:, 1] - y0) / res])).astype(np.int32)], 1)
        if mask.any():
            cands.append((q, mask.astype(bool)))
    covered = np.zeros((ny, nx), bool)
    chosen = []
    while cands and len(chosen) < max_views and covered.mean() < cover:
        gains = [int((m & ~covered).sum()) for _, m in cands]
        i = int(np.argmax(gains))
        if gains[i] == 0:
            break
        q, m = cands.pop(i)
        chosen.append(q)
        covered |= m
    chosen.sort(key=lambda q: float(q[0]))                              # sweep by the base (pan) joint
    return chosen, float(covered.mean())
