"""Measure a gripper from its URDF meshes: the bounding box of each hand link in the TCP frame (z = approach,
y = closing), at a given opening -> the `gripper.hand_boxes` of a robot profile (pnp_core/reorient.py uses them to
check that a hand turned to set an object on another face stays above the table).

    python3 /repo/tools/hand_geometry.py --robot so101 [--width 0.03]

For robots whose profile has a `description:` (the URDF we generate, pnp_core/robot_description.py). Collision
meshes, binary or ASCII STL, read with numpy.
"""
import argparse
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot, repo_root  # noqa: E402
from pnp_core.robot_description import urdf  # noqa: E402
from pnp_core.urdf_kin import Chain, _rpy  # noqa: E402


def read_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if data[:5].lower() == b"solid" and b"facet" in data[:400]:
        v = [ln.split()[1:4] for ln in data.decode(errors="replace").splitlines() if ln.strip().startswith("vertex")]
        return np.array(v, float)
    n = struct.unpack("<I", data[80:84])[0]
    tri = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return tri["v"].reshape(-1, 3).astype(float)


def jaw_angle(cfg, width):
    t = cfg["gripper"].get("width_table")
    return float(np.interp(width, t["width"], t["angle"])) if t else width / 2


def link_vertices(root, link):
    out = []
    el = root.find(f"link[@name='{link}']")
    for col in (el.findall("collision") if el is not None else []):
        m = col.find("geometry/mesh")
        if m is None:
            continue
        o = col.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = _rpy(*[float(x) for x in o.get("rpy", "0 0 0").split()])
            T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
        v = read_stl(Path(m.get("filename").replace("file://", "")))
        v = v * np.array([float(x) for x in m.get("scale", "1 1 1").split()])
        out.append(v @ T[:3, :3].T + T[:3, 3])
    return np.vstack(out) if out else np.zeros((0, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="so101")
    ap.add_argument("--width", type=float, default=0.03, help="opening [m] the jaws are measured at")
    ap.add_argument("--slices", action="store_true", help="the hand's extent band by band behind the tips")
    ap.add_argument("--bands-yaml", action="store_true", help="print gripper.hand_bands for the robot profile")
    a = ap.parse_args()
    cfg = load_robot(a.robot)
    xml = urdf(cfg, repo_root())
    root = ET.fromstring(xml)
    base, tcp = cfg["frames"]["base"], cfg["frames"]["tcp"]
    home = dict(zip(cfg["arm"]["joints"], cfg["arm"]["home"]))
    grip = {cfg["gripper"]["driven_joint"]: jaw_angle(cfg, a.width)}

    def T_of(link):
        ch = Chain(xml, base, link)
        return ch.fk([home.get(n, grip.get(n, 0.0)) for n in ch.names])
    T_tcp_inv = np.linalg.inv(T_of(tcp))
    print(f"{a.robot}: hand links in the TCP frame ({tcp}), jaws open {a.width * 1000:.0f} mm  [mm]")
    allv = []
    for link in cfg.get("hand_links", []):
        v = link_vertices(root, link)
        if not len(v):
            continue
        T = T_tcp_inv @ T_of(link)
        v = v @ T[:3, :3].T + T[:3, 3]
        allv.append(v)
        lo, hi = v.min(0), v.max(0)
        print(f"  {link:28s} x {lo[0] * 1000:7.1f} .. {hi[0] * 1000:6.1f}   y {lo[1] * 1000:7.1f} .. {hi[1] * 1000:6.1f}"
              f"   z {lo[2] * 1000:7.1f} .. {hi[2] * 1000:6.1f}")
    v = np.vstack(allv)
    if a.bands_yaml:
        # the hand as boxes, one per 10 mm band behind the tips, split into the part that doesn't move with the jaw
        # (gripper body, fixed jaw, wrist) and the moving jaw (the driven joint's child link): config gripper.hand_bands,
        # what pnp_core/pickability.py tests against the obstacle points (the moving boxes shift with the opening)
        j = root.find(f"joint[@name='{cfg['gripper']['driven_joint']}']")
        moving_link = j.find("child").get("link") if j is not None else None
        parts = {"fixed": [], "moving": []}
        for link in cfg.get("hand_links", []):
            w = link_vertices(root, link)
            if len(w):
                T = T_tcp_inv @ T_of(link)
                parts["moving" if link == moving_link else "fixed"].append(w @ T[:3, :3].T + T[:3, 3])
        print(f"  hand_bands:            # tools/hand_geometry.py --robot {a.robot} --width {a.width} --bands-yaml "
              f"(TCP frame [m]: [z_hi, z_lo, x_lo, x_hi, y_lo, y_hi])")
        print(f"    ref_width_m: {a.width}")
        for name, ws in parts.items():
            w = np.vstack(ws) if ws else np.zeros((0, 3))
            rows = []
            for z1 in np.arange(0.010, -0.131, -0.010):
                b = w[(w[:, 2] <= z1) & (w[:, 2] > z1 - 0.010)]
                if len(b):
                    rows.append([round(z1, 3), round(z1 - 0.010, 3), *(round(float(x), 4) for x in
                                 (b[:, 0].min(), b[:, 0].max(), b[:, 1].min(), b[:, 1].max()))])
            print(f"    {name}:")
            for r in rows:
                print(f"      - {r}")
    if a.slices:
        # where the hand is wide, band by band behind the jaw tips: what a box model must cover near the object
        # (pnp_core/pickability.py: a body box starting too far back missed gripper_link beside a brick)
        print("  bands behind the tips (all hand links)       x [mm]            y [mm]")
        for z1 in np.arange(0.010, -0.121, -0.010):
            b = v[(v[:, 2] <= z1) & (v[:, 2] > z1 - 0.010)]
            if len(b):
                print(f"    z {z1 * 1000:6.0f} .. {(z1 - 0.010) * 1000:5.0f}   {b[:, 0].min() * 1000:7.1f} .. "
                      f"{b[:, 0].max() * 1000:6.1f}   {b[:, 1].min() * 1000:7.1f} .. {b[:, 1].max() * 1000:6.1f}")
    near = v[v[:, 2] > -0.12]                      # what matters near the table: the last 12 cm behind the TCP
    lo, hi = near.min(0), near.max(0)
    print(f"  all within 12 cm of the TCP    x {lo[0] * 1000:7.1f} .. {hi[0] * 1000:6.1f}   y {lo[1] * 1000:7.1f} .. "
          f"{hi[1] * 1000:6.1f}   z {lo[2] * 1000:7.1f} .. {hi[2] * 1000:6.1f}")


if __name__ == "__main__":
    main()
