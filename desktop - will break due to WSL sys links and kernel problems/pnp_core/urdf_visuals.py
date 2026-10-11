"""A small URDF reader for VISUAL rigs (no physics): the links, joints and mesh visuals of a CAD export, and where each
link is for given joint values. Used by tools/fit_camera_mount.py (fit the D435i's stand to the calibrated camera) and
sim/scene.py (show the stand in Isaac). Pure numpy + xml, so Isaac's Python and the ROS container both import it.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .geometry import make_transform


def rpy_R(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_R(axis, a) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def origin_T(el) -> np.ndarray:
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return make_transform(rpy_R(*rpy), xyz)


def load_urdf_visuals(path):
    """-> (root link, joints {child: (parent, T_parent_joint, axis | None, type, name)},
    visuals {link: [(mesh path, T_link_visual, rgba | None)]}). Mesh paths relative to the URDF's folder."""
    path = Path(path)
    root = ET.fromstring(path.read_text())
    mats = {m.get("name"): [float(v) for v in m.find("color").get("rgba").split()] for m in root.findall("material")
            if m.find("color") is not None}
    joints = {}
    for j in root.findall("joint"):
        ax = j.find("axis")
        joints[j.find("child").get("link")] = (j.find("parent").get("link"), origin_T(j),
                                               [float(v) for v in ax.get("xyz").split()] if ax is not None else None,
                                               j.get("type"), j.get("name"))
    visuals = {}
    for link in root.findall("link"):
        for v in link.findall("visual"):
            m = v.find("geometry/mesh")
            if m is None:
                continue
            mat = v.find("material")
            name = mat.get("name") if mat is not None else ""
            rgba = mats.get(name)
            if rgba is None and mat is not None and mat.find("color") is not None:
                rgba = [float(c) for c in mat.find("color").get("rgba").split()]
            visuals.setdefault(link.get("name"), []).append(
                (path.parent / m.get("filename").replace("package://", "").split("/", 1)[-1]
                 if m.get("filename").startswith("package://") else path.parent / m.get("filename"),
                 origin_T(v), rgba))
    links = [lk.get("name") for lk in root.findall("link")]
    base = next(lk for lk in links if lk not in joints)
    return base, joints, visuals


def link_T(link, joints, q: dict | None = None) -> np.ndarray:
    """root -> link, with movable joints at q {joint name: value} (default 0 = the URDF's pose)."""
    q = q or {}
    T, chain = np.eye(4), []
    while link in joints:
        chain.append(joints[link])
        link = joints[link][0]
    for _parent, Tj, axis, typ, name in reversed(chain):
        T = T @ Tj
        if typ in ("revolute", "continuous") and axis is not None and name in q:
            T = T @ make_transform(axis_R(axis, q[name]), [0, 0, 0])
        elif typ == "prismatic" and axis is not None and name in q:
            T = T @ make_transform(np.eye(3), np.asarray(axis, float) * q[name])
    return T


def read_stl_triangles(path) -> np.ndarray:
    """(N, 3, 3) triangles of a binary or ASCII STL file."""
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [ln.split()[1:4] for ln in data.decode(errors="ignore").splitlines() if ln.strip().startswith("vertex")]
        return np.array(v, float).reshape(-1, 3, 3)
    n = int(np.frombuffer(data[80:84], np.uint32)[0])
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3, 3).astype(np.float64)
