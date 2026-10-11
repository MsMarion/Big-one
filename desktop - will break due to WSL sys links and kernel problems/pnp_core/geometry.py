"""Rotations, rigid transforms and camera frames.

Conventions (worth memorizing; most sim-to-real bugs live here):
  - Quaternions are (x, y, z, w), the ROS order. (Isaac Sim / USD APIs often use (w, x, y, z)!)
  - A 4x4 transform T_a_b maps points expressed in frame b into frame a:  p_a = T_a_b @ p_b.
  - ROS optical camera frame: +z forward (into the scene), +x right, +y down.
  - USD camera frame:         -z forward, +x right, +y up.
"""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero vector")
    return v / n


def quat_to_matrix(q) -> np.ndarray:
    """(x, y, z, w) -> 3x3 rotation matrix."""
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (x, y, z, w), w >= 0."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF / ROS roll-pitch-yaw (fixed axes: x, then y, then z) -> rotation matrix: Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to an (N, 3) array of points."""
    points = np.asarray(points, dtype=float)
    return points @ T[:3, :3].T + T[:3, 3]


def camera_mount_T(mount: dict) -> np.ndarray:
    """A fixed camera mount -> base <- optical 4x4: the full orientation `quat_xyzw` when given (a calibrated pose:
    keeps the camera's roll about its view axis), else `position` looking at `look_at` (no roll)."""
    if mount.get("quat_xyzw") is not None:
        return make_transform(quat_to_matrix(mount["quat_xyzw"]), mount["position"])
    return make_transform(look_at_optical(mount["position"], mount["look_at"]), mount["position"])


def look_at_optical(eye, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation (columns = axes in world) of a ROS *optical* camera at `eye` looking at `target`."""
    forward = normalize(np.asarray(target, float) - np.asarray(eye, float))
    right = normalize(np.cross(forward, np.asarray(world_up, float)))
    down = np.cross(forward, right)
    return np.column_stack([right, down, forward])


def optical_to_usd_camera(R_optical: np.ndarray) -> np.ndarray:
    """Same physical camera, USD axis convention: x right, y up, z backward (looks down -z)."""
    return R_optical @ np.diag([1.0, -1.0, -1.0])


# Rotation of the optical frame expressed in the camera BODY frame (REP-103 camera_link: x forward,
# y left, z up). This is the fixed camera_link -> *_optical_frame rotation used by realsense2_camera
# (rpy = -pi/2, 0, -pi/2): optical x(right) = -y_link, optical y(down) = -z_link, optical z(fwd) = x_link.
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def optical_to_body(R_base_optical: np.ndarray) -> np.ndarray:
    """Rotation of camera_link in the base frame, given the optical frame's rotation in the base frame."""
    return R_base_optical @ R_LINK_OPTICAL.T


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    return float(np.pi - (np.pi - a) % (2.0 * np.pi))
