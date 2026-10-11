"""Hand-in-the-workspace detection (pure logic; the ROS node is pnp_perception/hand_detector_node.py).

Two detectors, one output (pixel points of each hand found):
  sim_marker  the sim's stand-in hand (sim command hand_prop: a magenta box): an HSV threshold, the blob's pixels
  mediapipe   a real hand: MediaPipe's HandLandmarker (21 landmarks per hand) on the colour image
Then each hand's pixels + the aligned depth -> one base-frame point (median), and "in the workspace" = within
radius_m of the robot base (horizontally) and below max_height_m above the table. A hand with no valid depth counts
as IN the workspace (when unsure, pause).
"""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV HSV: H 0-179. The prop's displayColor (0.95, 0.05, 0.85) -> H ~152; anything else in the scene is far from it
MAGENTA_LO = np.array([140, 120, 70], np.uint8)
MAGENTA_HI = np.array([170, 255, 255], np.uint8)


def magenta_blobs(rgb: np.ndarray, min_area_px: int = 150) -> list[np.ndarray]:
    """-> one (N, 2) array of pixel (u, v) per magenta blob of at least min_area_px."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, MAGENTA_LO, MAGENTA_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] >= min_area_px:
            v, u = np.nonzero(labels == k)
            out.append(np.column_stack([u, v]))
    return out


def pixels_to_base(px: np.ndarray, depth_m: np.ndarray, K: np.ndarray, T_base_opt: np.ndarray, max_samples=400):
    """Median base-frame point of the pixels (N, 2) with valid depth -> (3,) or None (no valid depth)."""
    if len(px) > max_samples:
        px = px[np.linspace(0, len(px) - 1, max_samples).astype(int)]
    h, w = depth_m.shape
    u = np.clip(px[:, 0].round().astype(int), 0, w - 1)
    v = np.clip(px[:, 1].round().astype(int), 0, h - 1)
    d = depth_m[v, u]
    ok = (d > 0.05) & (d < 3.0)
    if ok.sum() < 3:
        return None
    u, v, d = u[ok], v[ok], d[ok]
    pc = np.column_stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d])
    pb = pc @ T_base_opt[:3, :3].T + T_base_opt[:3, 3]
    return np.median(pb, axis=0)


def in_workspace(p, table_z: float, radius_m: float, max_height_m: float) -> bool:
    """A hand point (base frame) inside the arm's space? None (no depth) -> True: when unsure, pause."""
    if p is None:
        return True
    return float(np.hypot(p[0], p[1])) <= radius_m and float(p[2]) <= table_z + max_height_m


class MediaPipeHands:
    """MediaPipe's HandLandmarker (Tasks API, mediapipe >= 0.10) -> per hand the 21 landmark pixels (N, 2).
    model: the hand_landmarker.task file (Google's, downloaded separately)."""

    def __init__(self, model_path: str, num_hands: int = 2, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        self.mp = mp
        opts = vision.HandLandmarkerOptions(base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                                            running_mode=vision.RunningMode.IMAGE, num_hands=num_hands,
                                            min_hand_detection_confidence=min_conf,
                                            min_hand_presence_confidence=min_conf)
        self.lm = vision.HandLandmarker.create_from_options(opts)

    def __call__(self, rgb: np.ndarray) -> list[np.ndarray]:
        h, w = rgb.shape[:2]
        res = self.lm.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        return [np.array([[p.x * w, p.y * h] for p in hand]) for hand in (res.hand_landmarks or [])]
