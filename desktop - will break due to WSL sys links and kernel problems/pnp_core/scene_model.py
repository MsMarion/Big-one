"""Build a metric model of the scene from the depth camera: which objects stand on the table, and how big are they?

    object points (base frame, already without robot / table / target cans; pnp_core.cloud)
      -> split into objects   (connected regions on a 1 cm grid seen from above)
      -> fit a box per object (minimum-area rectangle around the footprint, height = top surface above the table)
      -> compare with ground truth when there is one (the sim's box list): size / position / yaw errors in mm, deg

Why this gives METRIC sizes at all: a depth pixel is a distance in meters, and the camera intrinsics (fx, fy, cx,
cy) turn the pixel's row/column into the two other coordinates in meters (pnp_core.cloud.depth_to_cloud). The
camera pose (extrinsics: config/camera_d435i.yaml or the real calibration) puts them into the robot's frame. Each
link of that chain has its own error signature:
  - depth scale wrong by x %         -> every distance, hence every size, wrong by ~x %
  - fx/fy wrong by x %               -> sizes ACROSS the image wrong by ~x %, depth-direction sizes unaffected
  - camera pose tilted by 1 deg      -> positions off by ~1.5 cm at 0.9 m, sizes almost unaffected
That's why comparing measured sizes with known ones is a quick calibration check.

Pure numpy + OpenCV; unit-tested in tests/test_scene_model.py.
"""

from __future__ import annotations

import cv2
import numpy as np


def cluster_objects(points: np.ndarray, cell: float = 0.01, min_points: int = 30) -> list[np.ndarray]:
    """Split points into objects: rasterize their XY onto a `cell` grid, close 1-cell gaps, connected components."""
    if len(points) == 0:
        return []
    ij = np.floor(points[:, :2] / cell).astype(np.int64)
    lo = ij.min(axis=0)
    ij -= lo
    h, w = ij.max(axis=0) + 1
    grid = np.zeros((int(h) + 2, int(w) + 2), np.uint8)
    grid[ij[:, 0] + 1, ij[:, 1] + 1] = 1
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))     # bridge sparse-pixel gaps
    n, labels = cv2.connectedComponents(grid, connectivity=8)
    lab = labels[ij[:, 0] + 1, ij[:, 1] + 1]
    out = [points[lab == k] for k in range(1, n)]
    return [o for o in out if len(o) >= min_points]


def fit_box(points: np.ndarray, table_z: float, top_pct: float = 98.0) -> dict:
    """Oriented box standing on the table around an object's points: {center, size, yaw} (base frame, meters/rad).
    Footprint = minimum-area rectangle of the XY points (the camera sees the top face from above, so the footprint
    is complete even though the back sides are hidden); height = robust top (percentile) above the table."""
    (cx, cy), (w, l), ang = cv2.minAreaRect(points[:, :2].astype(np.float32))
    top = float(np.percentile(points[:, 2], top_pct))
    h = max(top - table_z, 0.0)
    yaw = float(np.deg2rad(ang))
    if w < l:                                    # convention: size[0] >= size[1], yaw = direction of size[0]
        w, l, yaw = l, w, yaw + np.pi / 2
    yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)   # a box is 180-deg symmetric
    return {"center": [float(cx), float(cy), table_z + h / 2], "size": [float(w), float(l), h], "yaw": yaw,
            "num_points": int(len(points))}


def _canonical(box: dict):
    """size sorted large -> small in XY, yaw of the larger side folded into (-90, 90] deg."""
    sx, sy, sz = box["size"]
    yaw = box["yaw"]
    if sx < sy:
        sx, sy, yaw = sy, sx, yaw + np.pi / 2
    return np.array([sx, sy, sz]), float((yaw + np.pi / 2) % np.pi - np.pi / 2)


def compare_boxes(measured: list[dict], truth: list[dict], max_dist: float = 0.10) -> list[dict]:
    """Match each true box with the nearest measured one (XY center distance < max_dist) and report errors.
    Result per true box: {id, matched, size_err_mm [dx, dy, dz], center_err_mm, yaw_err_deg}."""
    out, used = [], set()
    for t in truth:
        tc = np.asarray(t["center"][:2])
        best, bd = None, max_dist
        for i, m in enumerate(measured):
            d = float(np.linalg.norm(np.asarray(m["center"][:2]) - tc))
            if i not in used and d < bd:
                best, bd = i, d
        rec = {"id": t.get("id", "?"), "matched": best is not None}
        if best is not None:
            used.add(best)
            ts, ty = _canonical(t)
            ms, my = _canonical(measured[best])
            dyaw = (my - ty + np.pi / 2) % np.pi - np.pi / 2
            if ts[0] - ts[1] < 0.01:             # square footprint: yaw only defined modulo 90 deg
                dyaw = (dyaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
            rec.update(size_err_mm=np.round((ms - ts) * 1000, 1).tolist(),
                       center_err_mm=round(bd * 1000, 1),
                       yaw_err_deg=round(float(np.degrees(dyaw)), 1), measured_index=best)
        out.append(rec)
    return out
