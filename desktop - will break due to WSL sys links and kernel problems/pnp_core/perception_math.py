"""From a segmentation mask + aligned depth image to a 3D object pose.

Pipeline (see plan section 2a):
  depth image --depth_to_meters--> meters
  mask + depth --mask_to_points--> (N,3) points in the camera optical frame
  points --transform_points (geometry)--> points in the robot base frame
  points --estimate_cylinder / estimate_box--> object center (x, y), top height, yaw

Why not just average the points? The camera only sees the front and the top of the object, so
the average sits on the camera-facing surface, biased toward the camera by up to the object's
radius (~3 cm for a soup can) - enough to miss the grasp. We use the known object shape instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, K) -> "Intrinsics":
        """From a row-major 3x3 K (e.g. sensor_msgs/CameraInfo.k)."""
        K = np.asarray(K, dtype=float).reshape(3, 3)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    def project(self, points_cam: np.ndarray) -> np.ndarray:
        """(N,3) optical-frame points -> (N,2) pixel coords (u, v)."""
        p = np.asarray(points_cam, dtype=float)
        return np.column_stack([self.fx * p[:, 0] / p[:, 2] + self.cx, self.fy * p[:, 1] / p[:, 2] + self.cy])


@dataclass
class ObjectEstimate:
    center_xy: np.ndarray      # (2,) object axis / box center in the base frame [m]
    z_top: float               # top surface height in the base frame [m]
    yaw: float                 # rotation about base z [rad]; 0 for symmetric objects
    num_points: int
    fit_residual_m: float      # RMS residual of the shape fit (quality signal)
    height_m: float = 0.0      # the object's height as it rests (which face is down: a box on its side is lower)
    footprint_m: tuple = ()    # its (long, short) footprint as it rests


def depth_to_meters(depth: np.ndarray, encoding: str) -> np.ndarray:
    """16UC1 (millimeters, real RealSense) or 32FC1 (meters, Isaac default) -> float32 meters.
    Invalid pixels (0, NaN, inf) become 0."""
    if encoding in ("16UC1", "mono16"):
        d = depth.astype(np.float32) * 0.001
    elif encoding == "32FC1":
        d = depth.astype(np.float32)
    else:
        raise ValueError(f"unsupported depth encoding {encoding!r}")
    d[~np.isfinite(d)] = 0.0
    return d


def deproject(us: np.ndarray, vs: np.ndarray, z: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Pixels (u, v) with depth z (distance along the optical axis) -> (N,3) optical-frame points."""
    x = (us - K.cx) * z / K.fx
    y = (vs - K.cy) * z / K.fy
    return np.column_stack([x, y, z]).astype(np.float64)


def mask_to_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    K: Intrinsics,
    erode_px: int | None = None,
    min_depth: float = 0.1,
    max_depth: float = 3.0,
    mad_k: float = 3.5,
) -> np.ndarray:
    """Object mask + aligned depth -> (N,3) points in the camera optical frame.

    - Erode the mask: edge pixels mix object and background depth ("flying pixels"). By default (None) by up to 3 px,
      but at most a third of the mask's half-width: a 1x2 LEGO brick seen from 48 cm is ~15 x 8 px, and a fixed 3 px
      left 3 pixels of it (the can / box masks are 40+ px wide and keep the full 3 px).
    - Drop invalid / out-of-range depth.
    - Drop depth outliers: keep |z - median| < mad_k * 1.4826 * MAD (robust z-score).
    """
    m = mask.astype(np.uint8)
    if erode_px is None:
        half_w = float(cv2.distanceTransform(m, cv2.DIST_L2, 3).max()) if m.any() else 0.0
        erode_px = int(np.clip(half_w // 3, 1, 3))
    if erode_px > 0:
        m = cv2.erode(m, np.ones((2 * erode_px + 1, 2 * erode_px + 1), np.uint8))
    vs, us = np.nonzero(m)
    z = depth_m[vs, us]
    ok = (z > min_depth) & (z < max_depth)
    us, vs, z = us[ok], vs[ok], z[ok]
    if z.size == 0:
        return np.zeros((0, 3))
    med = np.median(z)
    mad = np.median(np.abs(z - med)) * 1.4826 + 1e-4
    ok = np.abs(z - med) < mad_k * mad
    return deproject(us[ok].astype(float), vs[ok].astype(float), z[ok], K)


def _top_height(points_base: np.ndarray, pct: float) -> float:
    return float(np.percentile(points_base[:, 2], pct))


def estimate_cylinder(
    points_base: np.ndarray,
    radius: float,
    camera_xy=None,
    top_pct: float = 95.0,
    iters: int = 20,
    side_margin_m: float = 0.01,
    min_side_points: int = 30,
) -> ObjectEstimate:
    """Upright cylinder with known radius: find the axis (cx, cy) minimizing
    sum_i (||p_i - c|| - r)^2 over the XY projection of the visible *side-wall* points (Gauss-Newton).

    Top-cap points project INSIDE the circle, not onto it, so they are excluded from the circle fit.
    If the camera sees (almost) only the cap - e.g. looking straight down - the cap centroid is
    already the axis, so we return that.

    Init: the visible wall points are on the camera-facing side, so start at their centroid pushed
    half a radius further away from the camera (if camera_xy is given)."""
    z_top = _top_height(points_base, top_pct)
    side = points_base[points_base[:, 2] < z_top - side_margin_m]
    if len(side) < min_side_points:
        cap = points_base[points_base[:, 2] >= z_top - side_margin_m]
        return ObjectEstimate(cap[:, :2].mean(axis=0), z_top, 0.0, len(points_base), 0.0)

    xy = side[:, :2]
    c = xy.mean(axis=0)
    if camera_xy is not None:
        away = c - np.asarray(camera_xy, dtype=float)
        n = np.linalg.norm(away)
        if n > 1e-9:
            c = c + radius * 0.5 * away / n
    for _ in range(iters):
        d = xy - c
        dist = np.linalg.norm(d, axis=1) + 1e-12
        res = dist - radius
        J = -d / dist[:, None]                     # d(res)/d(c)
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    res = np.linalg.norm(xy - c, axis=1) - radius
    return ObjectEstimate(c, z_top, 0.0, len(points_base), float(np.sqrt(np.mean(res**2))))


def estimate_box(points_base: np.ndarray, top_pct: float = 95.0) -> ObjectEstimate:
    """Box-like object: minimum-area rectangle around the XY projection -> center + yaw.
    Yaw is the rectangle's long-side direction wrapped to (-pi/2, pi/2] (a box has 180 deg symmetry).
    Note: with only 1-2 visible faces the rectangle can under-estimate the far side; a known-dims
    correction is a later improvement (see docs/learning/04_perception.md)."""
    xy = points_base[:, :2].astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(xy)
    yaw = np.deg2rad(angle_deg if w >= h else angle_deg + 90.0)
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return ObjectEstimate(np.array([cx, cy], float), _top_height(points_base, top_pct), float(yaw), len(points_base), 0.0)


@dataclass
class Quality:
    """Thresholds that turn a fit into accept/reject. A rejected estimate is reported as 'not found'
    rather than passed on as a confident-looking but wrong position."""
    min_points: int = 80              # fewer depth points than this: mostly occluded / too far / bad depth
    max_fit_residual_m: float = 0.006 # cylinder fit RMS: a can's wall is a circle; large residual = not a can-shape
    height_tolerance_m: float = 0.03  # top height must match the object's height on one of its faces


def estimate_object(mask, depth_m, K: Intrinsics, T_base_cam, obj_cfg: dict, table_z: float = 0.0,
                    quality: Quality = Quality()):
    """Segmentation mask + aligned depth -> (ObjectEstimate | None, reason).

    The whole perception geometry in one call (used by the ROS node and the tests):
      mask_to_points -> base frame -> shape fit (cylinder/box from the catalog) -> sanity checks.
    `obj_cfg` is the object's entry in config/objects.yaml (shape, dims_m)."""
    from .geometry import transform_points

    pts_cam = mask_to_points(mask, depth_m, K)
    if len(pts_cam) < quality.min_points:
        return None, f"too few depth points ({len(pts_cam)})"
    pts = transform_points(T_base_cam, pts_cam)
    dims = [float(v) for v in obj_cfg["dims_m"]]
    top = est_top = None
    if obj_cfg["shape"] == "cylinder":
        est = estimate_cylinder(pts, radius=dims[0] / 2, camera_xy=T_base_cam[:2, 3])
        est_top = est.z_top - table_z
        if abs(est_top - dims[2]) <= quality.height_tolerance_m:          # standing (or upside down)
            if est.fit_residual_m > quality.max_fit_residual_m:
                return None, f"cylinder fit residual {est.fit_residual_m * 1000:.1f} mm"
            est.height_m, est.footprint_m = dims[2], (dims[0], dims[0])
            return est, "ok"
        if abs(est_top - dims[0]) <= quality.height_tolerance_m:          # lying: from above it's a rectangle
            est = estimate_box(pts)
            est.height_m, est.footprint_m = dims[0], (dims[2], dims[0])
            return est, "ok (lying)"
        top = est_top
    else:
        # a box rests on any of its faces: its height is then one of its three sizes (pnp_core/reorient.py)
        est = estimate_box(pts)
        top = est.z_top - table_z
        k = int(np.argmin([abs(top - d) for d in dims]))
        if abs(top - dims[k]) <= quality.height_tolerance_m:
            rest = sorted((dims[i] for i in range(3) if i != k), reverse=True)
            est.height_m, est.footprint_m = dims[k], tuple(rest)
            return est, "ok" if k == 2 else "ok (on another face)"
    heights = sorted({round(d, 3) for d in (dims if obj_cfg["shape"] != "cylinder" else (dims[2], dims[0]))})
    return None, f"top height {top:.3f} m matches none of its faces ({', '.join(f'{h:.3f}' for h in heights)} m)"
