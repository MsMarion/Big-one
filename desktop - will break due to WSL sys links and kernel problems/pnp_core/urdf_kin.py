"""Kinematics of any serial arm straight from its URDF: forward kinematics, numerical IK, reach maps.

What this file teaches: a URDF is a tree of links joined by joints; each joint has a fixed `origin` (where its frame
sits on the parent link) and, if it moves, an `axis` it turns about. Forward kinematics is just multiplying
origin @ rotation(axis, q) down the chain. Inverse kinematics here is numerical (least squares within the joint
limits), slow but generic: it works for a 7-joint FR3 and a 5-joint SO-101 alike, which is the point
(docs/plans/platform_design.md). Pure numpy/scipy, no ROS.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


def _rpy(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, q) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


@dataclass
class Joint:
    name: str
    type: str                  # revolute | prismatic | fixed | continuous
    parent: str
    child: str
    T_origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


class Chain:
    """The serial chain base_link -> tip_link of a URDF."""

    def __init__(self, urdf_xml: str, base: str, tip: str):
        root = ET.fromstring(urdf_xml)
        by_child = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            ax = j.find("axis")
            lim = j.find("limit")
            lo, hi = (-np.pi, np.pi) if lim is None else (float(lim.get("lower", -np.pi)), float(lim.get("upper", np.pi)))
            by_child[j.find("child").get("link")] = Joint(
                j.get("name"), j.get("type"), j.find("parent").get("link"), j.find("child").get("link"), T,
                np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]), lo, hi)
        chain, link = [], tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"no joint chain from {base} to {tip} (stuck at {link})")
            chain.append(by_child[link])
            link = by_child[link].parent
        self.joints = chain[::-1]
        self.active = [j for j in self.joints if j.type in ("revolute", "continuous", "prismatic")]
        self.names = [j.name for j in self.active]
        self.lower = np.array([j.lower for j in self.active])
        self.upper = np.array([j.upper for j in self.active])

    @classmethod
    def from_file(cls, path, base, tip):
        with open(path, encoding="utf-8") as f:
            return cls(f.read(), base, tip)

    def fk(self, q) -> np.ndarray:
        """T_base_tip for joint values q (in self.names order)."""
        T, i = np.eye(4), 0
        for j in self.joints:
            T = T @ j.T_origin
            if j.type in ("revolute", "continuous"):
                M = np.eye(4)
                M[:3, :3] = _axis_angle(j.axis, q[i])
                T, i = T @ M, i + 1
            elif j.type == "prismatic":
                M = np.eye(4)
                M[:3, 3] = j.axis / np.linalg.norm(j.axis) * q[i]
                T, i = T @ M, i + 1
        return T

    def ik(self, target_pos, approach=None, q0=None, approach_weight=0.05, seeds=6, rng=None, target_R=None,
           rot_weight=0.1, tol_pos=1e-4, tol_deg=0.5):
        """Joint values that put the tip at target_pos [m] and, optionally,
          approach   its z axis along this unit vector (e.g. [0, 0, -1] = straight down; rotation about it free), or
          target_R   its full orientation (3x3): a 5-joint arm can still match a top-down pose at any yaw.
        Least squares within the joint limits, from q0 (the current joints: stays on the same arm "branch") then
        random seeds. -> (q, position error [m], orientation error [deg]) of the best seed, or (None, inf, inf)."""
        from scipy.optimize import least_squares
        rng = np.random.default_rng(0) if rng is None else rng
        target_pos = np.asarray(target_pos, float)
        a = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
        Rt = None if target_R is None else np.asarray(target_R, float)

        def res(q):
            T = self.fk(q)
            r = T[:3, 3] - target_pos
            if Rt is not None:
                return np.concatenate([r, rot_weight * (T[:3, :3] - Rt).ravel()])
            return r if a is None else np.concatenate([r, approach_weight * (T[:3, 2] - a)])

        def ang(T):
            if Rt is not None:
                return float(np.degrees(np.arccos(np.clip((np.trace(Rt.T @ T[:3, :3]) - 1) / 2, -1, 1))))
            return 0.0 if a is None else float(np.degrees(np.arccos(np.clip(T[:3, 2] @ a, -1, 1))))

        best = (None, np.inf, np.inf)
        lo, hi = self.lower + 1e-6, self.upper - 1e-6
        starts = ([np.clip(q0, lo, hi)] if q0 is not None else []) + \
                 [rng.uniform(lo, hi) for _ in range(seeds)] + [np.clip(np.zeros(len(lo)), lo, hi)]
        for s in starts:
            sol = least_squares(res, s, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=400)
            T = self.fk(sol.x)
            pe, ae = float(np.linalg.norm(T[:3, 3] - target_pos)), ang(T)
            if pe + ae / 1000 < best[1] + best[2] / 1000:
                best = (sol.x, pe, ae)
            if pe < tol_pos and ae < tol_deg:
                break
        return best

    def clamp_limits(self, limits: dict, margin: float = 0.0):
        """Narrow the joint limits to a robot profile's position_limits (minus margin), like the launch does."""
        for i, n in enumerate(self.names):
            if n in limits:
                self.lower[i] = max(self.lower[i], limits[n][0] + margin)
                self.upper[i] = min(self.upper[i], limits[n][1] - margin)
        return self


def reach_map(chain: Chain, xs, ys, z, approach=(0, 0, -1), pos_tol=0.002, ang_tol_deg=3.0):
    """Which (x, y) points at height z the tip can reach with the given approach direction.
    -> boolean grid [len(ys), len(xs)] and the IK solutions (None where unreachable)."""
    ok = np.zeros((len(ys), len(xs)), bool)
    sols = [[None] * len(xs) for _ in ys]
    q_prev = None
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            q, pe, ae = chain.ik([x, y, z], approach, q0=q_prev, seeds=3)
            if q is not None and pe < pos_tol and ae < ang_tol_deg:
                ok[iy, ix], sols[iy][ix], q_prev = True, q, q
    return ok, sols
