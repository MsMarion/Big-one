"""The AprilTag layout of a cell: where each tag lies on the board (relative to the robot base), which the fixed
camera sees, and a print-ready layout sheet to place the real ones with a tape measure.

    python3 /repo/tools/tag_layout.py [scene]          visibility table (default scene: so101_plate_tags)
    python3 /repo/tools/tag_layout.py [scene] --pdf    + data/print/<cell>_tag_layout.pdf (to-scale top view, a table
                                                         in cm AND inches, each tag at 100 % with a check bar)

The tags come from the scene's calibration_targets file (apriltags.tags[].sim_xy, sim_yaw_deg; base frame: x forward,
y left, the base_link origin on the board). The camera: the scene's believed mount (camera.mount, incl.
from_calibration) and intrinsics. A tag counts as SEEN when all four outer corners (with its white border) are at
least `margin` px inside the image.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_fixed_camera, load_yaml  # noqa: E402
from pnp_core.geometry import camera_mount_T  # noqa: E402

M_TO_IN = 39.3701


def scene_of(name):
    p = Path(name) if name.endswith(".yaml") else Path(f"/repo/config/scenes/{name}.yaml")
    return load_yaml(str(p)), p


def tag_outline(xy, yaw_deg, size, z):
    """The 4 corners of a tag lying flat (with a 25 % white border each side: the printed square)."""
    h = size * 1.5 / 2
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ R.T + np.asarray(xy)
    return np.column_stack([pts, np.full(4, z)])


def project(T_base_opt, K, pts):
    p = (pts - T_base_opt[:3, 3]) @ T_base_opt[:3, :3]          # base -> optical
    if np.any(p[:, 2] <= 0.01):
        return None, p[:, 2]
    uv = (p[:, :2] / p[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    return uv, p[:, 2]


def visibility(scene, margin=8):
    cam = load_fixed_camera(scene)
    T = camera_mount_T(cam["mount"])
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    W, H = cam["width"], cam["height"]
    t = scene["table"]
    z = t["center"][2] + t["size"][2] / 2
    a = load_yaml(f"/repo/{scene['calibration_targets']}")["apriltags"]
    rows = []
    for tag in a["tags"]:
        uv, depth = project(T, K, tag_outline(tag["sim_xy"], tag.get("sim_yaw_deg", 0.0), a["size_m"], z))
        seen = uv is not None and bool(np.all((uv[:, 0] >= margin) & (uv[:, 0] <= W - margin)
                                              & (uv[:, 1] >= margin) & (uv[:, 1] <= H - margin)))
        side_px = float(np.mean(np.linalg.norm(np.roll(uv, -1, 0) - uv, axis=1))) / 1.5 if uv is not None else 0.0
        x, y = tag["sim_xy"]
        rows.append({"id": tag["id"], "where": tag.get("where", ""), "x": x, "y": y, "yaw": tag.get("sim_yaw_deg", 0.0),
                     "dist": math.hypot(x, y), "angle": math.degrees(math.atan2(y, x)), "seen": seen,
                     "px": side_px, "range_m": float(np.mean(depth)), "center_uv": None if uv is None else uv.mean(0)})
    return rows, a, (T, K, W, H, z)


def print_table(rows):
    print(f"{'id':>3} {'where':22s} {'x cm':>7} {'y cm':>7} {'x in':>6} {'y in':>6} {'dist cm/in':>12} {'angle':>6} "
          f"{'seen':>5} {'tag px':>6} {'range':>6}")
    for r in rows:
        print(f"{r['id']:>3} {r['where']:22s} {r['x'] * 100:7.1f} {r['y'] * 100:7.1f} {r['x'] * M_TO_IN:6.1f} "
              f"{r['y'] * M_TO_IN:6.1f} {r['dist'] * 100:6.1f}/{r['dist'] * M_TO_IN:4.1f}\" {r['angle']:5.0f}° "
              f"{'YES' if r['seen'] else 'no':>5} {r['px']:6.0f} {r['range_m']:5.2f}m")
    print(f"{sum(r['seen'] for r in rows)} of {len(rows)} tags fully in the fixed camera's view")


def footprint(T, K, W, H, z):
    """Where the image's corners land on the board (base frame): what the camera covers."""
    out = []
    for u, v in ((0, 0), (W, 0), (W, H), (0, H)):
        d = T[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        s = (z - T[2, 3]) / d[2] if abs(d[2]) > 1e-9 else np.inf
        out.append(T[:3, 3] + s * d if s > 0 else None)
    return out


def pdf(scene, rows, a, geom, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Polygon, Rectangle
    t = scene["table"]
    cx, cy, sx, sy = t["center"][0], t["center"][1], t["size"][0], t["size"][1]
    T, K, W, H, z = geom
    with PdfPages(path) as pp:
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.add_patch(Rectangle((cy - sy / 2, cx - sx / 2), sy, sx, fill=False, lw=1.5))
        for g in np.arange(-0.5, 1.0, 0.05):                     # 5 cm grid, origin at the base
            ax.axhline(g, color="0.85", lw=0.4)
            ax.axvline(g, color="0.85", lw=0.4)
        fp = [p for p in footprint(T, K, W, H, z) if p is not None]
        if len(fp) == 4:
            ax.add_patch(Polygon([(p[1], p[0]) for p in fp], closed=True, fill=True, alpha=0.08, color="tab:blue",
                                 label="fixed camera's view"))
        ax.plot(0, 0, "ks", ms=10)
        ax.annotate("robot base (0, 0)", (0, 0), (0.02, -0.04), fontsize=8)
        cam = T[:3, 3]
        ax.plot(cam[1], cam[0], "b^", ms=9)
        ax.annotate(f"D435i ({cam[0] * 100:.0f}, {cam[1] * 100:.0f}) cm, {cam[2] * 100:.0f} cm up", (cam[1], cam[0]),
                    (cam[1] + 0.02, cam[0] + 0.02), fontsize=7, color="b")
        for r in rows:
            o = tag_outline((r["x"], r["y"]), r["yaw"], a["size_m"], 0)[:, :2]
            ax.add_patch(Polygon([(p[1], p[0]) for p in o], closed=True, fill=True,
                                 color="green" if r["seen"] else "orange"))
            ax.annotate(f"{r['id']}: x {r['x'] * 100:.1f} y {r['y'] * 100:.1f} cm", (r["y"], r["x"]),
                        (r["y"] + 0.025, r["x"] + 0.02), fontsize=7)
        ax.set_xlim(cy + sy / 2 + 0.05, cy - sy / 2 - 0.05)         # y LEFT is left on the page (looking from above,
        ax.set_ylim(cx - sx / 2 - 0.05, cx + sx / 2 + 0.05)         # robot at the bottom, x forward = up)
        ax.set_aspect("equal")
        ax.set_xlabel("y [m] (left of the robot +)")
        ax.set_ylabel("x [m] (in front of the robot +)")
        ax.set_title(f"{scene['name']}: AprilTag layout (top view, 5 cm grid)\ngreen = seen by the fixed camera, "
                     f"orange = not; board {sx * 100:.1f} x {sy * 100:.1f} cm", fontsize=10)
        ax.legend(loc="lower right", fontsize=7)
        pp.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        head = ["id", "where", "x cm", "y cm", "x in", "y in", "dist cm", "dist in", "angle", "yaw", "cam"]
        cells = [[r["id"], r["where"], f"{r['x'] * 100:.1f}", f"{r['y'] * 100:.1f}", f"{r['x'] * M_TO_IN:.2f}",
                  f"{r['y'] * M_TO_IN:.2f}", f"{r['dist'] * 100:.1f}", f"{r['dist'] * M_TO_IN:.2f}",
                  f"{r['angle']:.0f}°", f"{r['yaw']:.0f}°", "seen" if r["seen"] else "-"] for r in rows]
        tb = ax.table(cellText=cells, colLabels=head, loc="upper center", cellLoc="center")
        tb.auto_set_font_size(False)
        tb.set_fontsize(7)
        tb.auto_set_column_width(list(range(len(head))))
        tb.scale(1, 1.4)
        ax.set_title("Tag centres from the ROBOT BASE origin (base_link, under the shoulder joint): x forward, y to "
                     "the robot's left.\nMeasure on the board with a tape; each tag's printed square is "
                     f"{a['size_m'] * 1.5 * 100:.1f} cm (black {a['size_m'] * 100:.1f} cm), printed edge parallel "
                     "to x at yaw 0.", fontsize=8)
        pp.savefig(fig)
        plt.close(fig)
    print(f"{path}: layout (top view + table); the tags themselves: tools/make_calibration_targets.py --print "
          f"{scene.get('cell', scene['name'])}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    scene, _ = scene_of(args[0] if args else "so101_plate_tags")
    rows, a, geom = visibility(scene)
    print_table(rows)
    fp = footprint(*geom)
    print("fixed camera's view on the board (base frame, cm): "
          + "  ".join("beyond" if p is None else f"({p[0] * 100:.0f}, {p[1] * 100:.0f})" for p in fp))
    if "--pdf" in sys.argv:
        out = Path("/repo/data/print")
        out.mkdir(parents=True, exist_ok=True)
        pdf(scene, rows, a, geom, out / f"{scene.get('cell', scene['name'])}_tag_layout.pdf")


if __name__ == "__main__":
    main()
