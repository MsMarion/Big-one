"""Small URDF edits done at launch time (pure Python, unit-tested in tests/test_urdf_tools.py)."""

from __future__ import annotations

import xml.etree.ElementTree as ET


def clamp_joint_limits(urdf: str, limits: dict, margin: float = 0.0) -> tuple[str, list[str]]:
    """Narrow <limit lower/upper> (and the safety_controller soft limits) of the given joints to
    [lo + margin, hi - margin]. Never widens a limit. Returns (new urdf, list of human-readable changes)."""
    root = ET.fromstring(urdf)
    changes = []
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name not in limits:
            continue
        lo, hi = limits[name][0] + margin, limits[name][1] - margin
        lim = joint.find("limit")
        if lim is None:
            continue
        old = (float(lim.get("lower")), float(lim.get("upper")))
        new = (max(old[0], lo), min(old[1], hi))
        if new != old:
            lim.set("lower", f"{new[0]:.4f}")
            lim.set("upper", f"{new[1]:.4f}")
            changes.append(f"{name}: [{old[0]:.4f}, {old[1]:.4f}] -> [{new[0]:.4f}, {new[1]:.4f}]")
        sc = joint.find("safety_controller")
        if sc is not None and sc.get("soft_lower_limit") is not None:
            sc.set("soft_lower_limit", f"{max(float(sc.get('soft_lower_limit')), new[0]):.4f}")
            sc.set("soft_upper_limit", f"{min(float(sc.get('soft_upper_limit')), new[1]):.4f}")
    return ET.tostring(root, encoding="unicode"), changes
