"""Grasp study: many candidate grasps of an object, checked against a gripper, scored (docs/learning/20_grasp_study.md).

What this file teaches: "how should this robot hold this object?" has more answers than "straight down at the
center". A parallel gripper grasp is fixed by
    which side the fingers close across  (the closing axis: that side must fit the opening),
    where the hand comes from            (the approach: straight down, or tilted towards the side),
    how far in                           (depth: the fingers' length limits it, the palm must stay out),
    where along the object               (offset: off-center grasps raise the hand above the table when turning the
                                          object over, but gravity twists the object round the fingers more).
This module generates those combinations for an object resting on a face, drops the ones the GRIPPER can't do
(opening, finger length, hand below the table: the measured hand boxes), and gives each the scores that don't need
the arm: gravity twist about the closing axis, opening margin, tilt. Whether the ARM reaches a grasp across the
work area -- where robots really differ -- is tools/grasp_study.py (IK over a grid of object positions).

Frames: TCP z = approach (into the object), y = closing direction, x = y cross z (palm normal); same as the grasps of
pnp_core/object_grasp.py. Grasps are stored in the OBJECT frame (T_obj_tcp), so they follow the object anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .object_grasp import object_extents, up_axis
from .reorient import Rest, hand_corners, rest_poses

G = 9.81


@dataclass
class Candidate:
    rest: str                      # the face the object rests on
    closing: str                   # "x" / "y" / "z" of the object frame, or "diameter" (cylinder)
    tilt_deg: float                # approach from vertical (0 = straight down, 90 = horizontal)
    toward: int                    # +1 / -1: which way it tilts
    depth_m: float                 # fingertips this far into the object from where the approach enters it
    offset_m: float                # along the palm normal (TCP x) from the object's center line
    width_m: float                 # finger opening when holding it
    T_obj_tcp: np.ndarray          # the grasp in the object frame
    twist_nm: float = 0.0          # gravity torque about the closing axis while held as picked [N m]
    margin_m: float = 0.0          # gripper max opening - width
    scores: dict = field(default_factory=dict)      # filled by the reach study

    @property
    def name(self) -> str:
        return (f"across {self.width_m * 1000:.0f} mm, tilt {self.tilt_deg:.0f}{'+' if self.toward > 0 else '-'}, "
                f"{self.depth_m * 1000:.0f} mm in, {self.offset_m * 1000:+.0f} mm off-center")


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _support(d, axes, ext):
    """Half-extent of a box with these axes (columns) / extents along direction d."""
    return float(sum(abs(d @ axes[:, i]) * ext[i] / 2 for i in range(3)))


def finger_length(gripper_cfg: dict) -> float:
    """How deep the fingers reach in front of the palm: the finger box's extent from the TCP back (hand_boxes)."""
    for b in gripper_cfg.get("hand_boxes", []):
        if b.get("fingers"):
            return float(-(b["center"][2] - b["half"][2]))
    return 0.04


def candidates(obj_cfg: dict, gripper_cfg: dict, rest_name: str = "upright", tilts=(0, 15, 30, 45, 60, 75, 90),
               depths=(0.2, 0.5, 0.9), offsets=(0.0, 0.25, 0.4), table_z: float = 0.0, mass_kg: float | None = None):
    """Grasps of the object resting on `rest_name` (at yaw 0, centered at x = y = 0) that the gripper can make.
    depths: fractions of how far the fingers can go in (the shorter of their length and the object along the
    approach); offsets: fractions of the object's size along the palm normal."""
    rest: Rest = rest_poses(obj_cfg)[rest_name]
    R = rest.R
    center = np.array([0.0, 0.0, table_z + rest.height_m])
    m = float(mass_kg if mass_kg is not None else obj_cfg.get("mass_kg", 0.3))
    maxw = float(gripper_cfg["max_width_m"]) - 2 * float(gripper_cfg.get("open_margin_m", 0.0)) * 0.5
    flen = finger_length(gripper_cfg)
    ext = object_extents(obj_cfg)
    z = np.array([0.0, 0.0, 1.0])
    # closing axes: box -> each object axis whose size fits; cylinder -> across the diameter (perpendicular to the axis)
    closings = []
    if obj_cfg["shape"] == "cylinder":
        u = R @ up_axis(obj_cfg)
        dia = float(obj_cfg["dims_m"][0])
        if dia <= maxw:
            c = _unit(np.cross(u, z)) if abs(u[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            closings.append(("diameter", c, dia))
    else:
        for i, nm in enumerate("xyz"):
            if ext[i] <= maxw:
                closings.append((nm, R[:, i], float(ext[i])))
    out = []
    for nm, c, width in closings:
        if abs(c[2]) > 0.7:                       # closing vertically: one finger would have to be under the object
            continue
        c = _unit(c - c[2] * z)
        h = _unit(np.cross(c, z))
        for tilt in tilts:
            for s in ((1,) if tilt == 0 else (1, -1)):
                d = _unit(np.cos(np.radians(tilt)) * -z + np.sin(np.radians(tilt)) * s * h)
                x = _unit(np.cross(c, d))
                R_tcp = np.column_stack([x, c, d])
                # how far the object reaches along the approach from its center, and across the palm normal
                if obj_cfg["shape"] == "cylinder":
                    u = R @ up_axis(obj_cfg)
                    r, hh = obj_cfg["dims_m"][0] / 2, obj_cfg["dims_m"][2] / 2
                    sup = lambda v: abs(v @ u) * hh + r * np.sqrt(max(0.0, 1 - (v @ u) ** 2))   # noqa: E731
                    hd, hx = sup(d), sup(x)
                else:
                    hd, hx = _support(d, R, ext), _support(x, R, ext)
                for df in depths:
                    depth = df * min(flen, 2 * hd)
                    for of in offsets:
                        for so in ((1,) if of == 0 else (1, -1)):
                            off = so * min(of * 2 * hx, max(0.0, hx - 0.01))
                            p = center + off * x - (hd - depth) * d
                            T = np.eye(4)
                            T[:3, :3], T[:3, 3] = R_tcp, p
                            corners = hand_corners(gripper_cfg, width) @ R_tcp.T + p
                            if float(corners[:, 2].min()) < table_z + 0.002:          # hand into the table
                                continue
                            T_obj = np.eye(4)
                            T_obj[:3, :3], T_obj[:3, 3] = R, center
                            r_com = center - p
                            twist = abs(float(c @ np.cross(r_com, [0.0, 0.0, -m * G])))
                            out.append(Candidate(rest_name, nm, float(tilt), s, float(depth), float(off), width,
                                                 np.linalg.inv(T_obj) @ T, twist, float(maxw - width)))
    return dedupe(out)


def dedupe(cands):
    """Offsets clip to the object's size: on a small object several requested offsets give the same grasp."""
    seen, out = set(), []
    for c in cands:
        k = (c.rest, c.closing, c.tilt_deg, c.toward if c.tilt_deg else 0, round(c.depth_m, 4), round(c.offset_m, 4))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def static_rank(cands):
    """Best first by what needs no arm: little gravity twist, more opening margin, closer to straight down."""
    return sorted(cands, key=lambda k: (round(k.twist_nm, 3), -round(k.margin_m, 3), k.tilt_deg))
