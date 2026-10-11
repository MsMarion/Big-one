"""Score grasp candidates instead of taking the first that plans (pick_place_node, parameter grasp_scoring).

A candidate = one closing side of the object x one hand direction (for the SO-101 the two directions are different
grasps: the fixed jaw is on one side or the other). Per candidate:
  clearance     how far the nearest obstacle point is from the open hand's volumes (pnp_core/pickability.py) [m]
  roll_pick     the wrist roll at the grasp (numeric IK from home) [rad]; roll_place: at the place pose
  joint_margin  the smallest distance of any joint to its limit, at the grasp and at the place [rad]
  yaw_tol       how far the object's yaw may be off before a corner reaches a jaw [deg] (the close look still
                turns the estimate 3-6 deg: below ~6 deg the jaw lands on the brick)
  narrow        the narrowest grip's width / this one's (1 = across the narrow side): the old narrowest-first rule,
                weighted most -- a 2x4 grabbed across its 31.8 mm length for 6 mm more clearance tipped over (demo, 03:27)
score = 2 x narrow + clearance (capped 1 cm: more room doesn't help) + joint margin (capped 0.5 rad) + yaw tolerance
(capped 12 deg), each 0..1, minus
3 (more than the best score) for a candidate that already failed in this task: a retry tries a different one.
No IK at the grasp or the place
-> rejected (the place: the far-corner failure, a rotated goal the wrist couldn't turn to with that grasp).
"""
from __future__ import annotations

import math

import numpy as np

W_NARROW, W_CLEAR, W_MARGIN, W_YAW = 2.0, 1.0, 1.0, 0.5
CLEAR_CAP = 0.01
TRIED_PENALTY = 5.0          # more than any score: a retry always takes an untried candidate if there is one


def yaw_tolerance_deg(width: float, length: float, room: float) -> float:
    """The largest yaw error [deg] at which the object's corner still fits the room beside it: a box of `width`
    across the jaws and `length` along them, `room` between its face and the nearer jaw. Its half extent across the
    jaws at yaw error t: (length/2) sin t + (width/2) cos t <= width/2 + room."""
    for i in range(1, 181):
        t = math.radians(i * 0.25)
        if (length / 2) * math.sin(t) + (width / 2) * math.cos(t) > width / 2 + room:
            return (i - 1) * 0.25
    return 45.0


def joint_margin(q, joints, limits: dict) -> float:
    """The smallest distance of any joint to its limit [rad] (inf without limits)."""
    m = math.inf
    for name, v in zip(joints, q):
        if name in limits:
            lo, hi = limits[name]
            m = min(m, float(v) - lo, hi - float(v))
    return m


def score(c: dict) -> float:
    """c: {narrow, clearance, joint_margin, yaw_tol, tried} -> score (higher is better)."""
    s = W_NARROW * float(c.get("narrow", 1.0))
    s += W_CLEAR * min(max(c.get("clearance") or 0.0, 0.0), CLEAR_CAP) / CLEAR_CAP
    s += W_MARGIN * min(max(c.get("joint_margin") or 0.0, 0.0), 0.5) / 0.5
    s += W_YAW * min(c.get("yaw_tol") or 0.0, 12.0) / 12.0
    return s - (TRIED_PENALTY if c.get("tried") else 0.0)


def closing_side(T_base_tcp, T_base_obj) -> str:
    """Which object axis the jaws close across, and on which side the hand's +y (the fixed jaw, SO-101) is."""
    y_obj = T_base_obj[:3, :3].T @ T_base_tcp[:3, 1]        # the hand's closing direction in the object frame
    k = int(np.argmax(np.abs(y_obj[:2])))
    return f"{'xy'[k]}{'+' if y_obj[k] > 0 else '-'}"


def tried_match(yaw: float, tried, tol_deg: float = 20.0) -> bool:
    return any(abs(math.remainder(yaw - t, 2 * math.pi)) < math.radians(tol_deg) for t in tried)
