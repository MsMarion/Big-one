"""A to-scale map of the plate cell (real = twin, config/scenes/so101_plate.yaml): where the robot, the cameras, what
each camera covers on the plate, the reach band, the targets. Top view + side view.
    python3 tools/twin_map.py [--pose 0 0 0 80 0]  -> data/twin/map.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pnp_core import robot_description as rdesc  # noqa: E402
from pnp_core.catalog import load_fixed_camera, load_robot, load_scene, load_yaml, repo_root  # noqa: E402
from pnp_core.geometry import camera_mount_T, make_transform, quat_to_matrix  # noqa: E402
from pnp_core.urdf_kin import Chain  # noqa: E402

LINKS = ["shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "so101_tcp"]


def footprint(T, K, w, h, z):
    """The image corners' rays hit the plane z -> 4 points (or None where a ray misses)."""
    out = []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        d = T[:3, :3] @ np.linalg.solve(K, [u, v, 1.0])
        if d[2] >= -1e-6:
            return None
        out.append(T[:3, 3] + d * (z - T[2, 3]) / d[2])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", type=float, nargs=5, default=[0, 0, 0, 80, 0], help="arm joints (deg): the look pose")
    a = ap.parse_args()
    scene = load_scene("config/scenes/so101_plate.yaml")
    t = scene["table"]
    top = t["center"][2] + t["size"][2] / 2
    robot = load_robot("so101")
    off = load_yaml("config/calibration/so101_plate/joint_offsets_real.yaml")["offsets_deg"]
    urdf = rdesc.urdf(robot, repo_root(), "mock", offsets={j: np.radians(v) for j, v in off.items()})
    q = np.radians(a.pose)
    base = robot["frames"]["base"]
    pts = [np.zeros(3)]
    for link in LINKS:
        ch = Chain(urdf, base, link)
        pts.append(ch.fk(q[:len(ch.names)])[:3, 3])
    arm = np.array(pts)
    cam = load_fixed_camera(scene)
    T435 = camera_mount_T(cam["mount"])
    K435 = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
    wi = scene["wrist_camera_intrinsics"]
    K405 = np.array([[wi["fx"], 0, wi["cx"]], [0, wi["fy"], wi["cy"]], [0, 0, 1.0]])
    cal = load_yaml("config/calibration/so101_plate/real_d405.yaml")["T_parent_optical"]
    T405 = Chain(urdf, base, "gripper_link").fk(q) @ make_transform(quat_to_matrix(cal["quat_xyzw"]), cal["xyz"])
    f435 = footprint(T435, K435, 640, 480, top)
    f405 = footprint(T405, K405, 640, 360, top)
    bp = REPO / "data" / "twin" / "board_pose.json"

    fig, (ax, sx) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    # --- top view (x forward from the base, y to its left; drawn with x up like the D435i's view of the plate) ---
    def P(p):                       # base (x, y) -> plot (y, x) on a REVERSED y axis: +y (the robot's left) on the left
        p = np.atleast_2d(p)
        return p[:, 1], p[:, 0]
    cx_, cy_ = t["center"][:2]
    sxh, syh = t["size"][0] / 2, t["size"][1] / 2
    plate = np.array([[cx_ - sxh, cy_ - syh], [cx_ + sxh, cy_ - syh], [cx_ + sxh, cy_ + syh], [cx_ - sxh, cy_ + syh],
                      [cx_ - sxh, cy_ - syh]])
    ax.fill(*P(plate), color="#d9c09a", alpha=0.35, label="plate (4' x 4')")
    th = np.linspace(-np.pi / 2, np.pi / 2, 100)
    r0, r1 = scene["ui"]["reach_m"]
    band = np.r_[np.c_[r1 * np.cos(th), r1 * np.sin(th)], np.c_[r0 * np.cos(th[::-1]), r0 * np.sin(th[::-1])]]
    ax.fill(*P(band), color="#7fb3d5", alpha=0.35, label=f"reach band {r0:.2f}-{r1:.2f} m")
    sr = scene["target"]["spawn_region"]
    box = np.array([[sr["x"][0], sr["y"][0]], [sr["x"][1], sr["y"][0]], [sr["x"][1], sr["y"][1]], [sr["x"][0], sr["y"][1]],
                    [sr["x"][0], sr["y"][0]]])
    ax.plot(*P(box), "b-", lw=1.5, label="pick / place area")
    if f435 is not None:
        ax.fill(*P(np.vstack([f435[:, :2], f435[:1, :2]])), color="orange", alpha=0.18)
        ax.plot(*P(np.vstack([f435[:, :2], f435[:1, :2]])), color="orange", lw=1.5, label="D435i sees (on the plate)")
    if f405 is not None:
        ax.fill(*P(np.vstack([f405[:, :2], f405[:1, :2]])), color="green", alpha=0.18)
        ax.plot(*P(np.vstack([f405[:, :2], f405[:1, :2]])), color="green", lw=1.5, label="D405 sees (look pose)")
    ax.plot(*P(arm[:, :2]), "k-o", ms=3, lw=2.5, label="arm (look pose)")
    ax.plot(*P(T435[:3, 3][:2]), "s", color="darkorange", ms=11, label="D435i")
    ax.plot(*P(T405[:3, 3][:2]), "^", color="darkgreen", ms=10, label="D405 (on the wrist)")
    for k in scene["obstacles"].get("keep_clear", []):
        ax.plot(*P(k["xy"]), "kx", ms=10, mew=2)
        ax.annotate("AprilTag", P(k["xy"]), xytext=(6, -12), textcoords="offset points", fontsize=8)
    if bp.exists():
        T = np.array(json.loads(bp.read_text())["T_base_board"])
        c = (T @ np.array([0.0875, 0.07, 0, 1]))[:3]
        ax.plot(*P(c[:2]), "m+", ms=12, mew=2, label="hand board (real)")
    ax.set_aspect("equal")
    ax.set_xlim(0.45, -0.45)
    ax.set_ylim(-0.12, 0.70)
    ax.set_xlabel("y (m)  <- robot's left          right ->")
    ax.set_ylabel("x (m), forward from the base")
    ax.set_title("Top view (to scale)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")
    # --- side view (x, z) ---
    sx.fill_between([-0.1, 0.7], top - 0.0127, top, color="#d9c09a", alpha=0.6, label=f"plate (top z {top * 1000:.1f} mm)")
    sx.plot(arm[:, 0], arm[:, 2], "k-o", ms=3, lw=2.5, label="arm (look pose)")
    for T, col, K, (w, hh), nm in ((T435, "darkorange", K435, (640, 480), "D435i"), (T405, "darkgreen", K405, (640, 360), "D405")):
        p = T[:3, 3]
        sx.plot(p[0], p[2], "s" if nm == "D435i" else "^", color=col, ms=10, label=nm)
        for v in (0, hh):                      # the top and bottom edge rays (image middle column)
            d = T[:3, :3] @ np.linalg.solve(K, [w / 2, v, 1.0])
            s = (top - p[2]) / d[2] if d[2] < 0 else 0.4
            sx.plot([p[0], p[0] + d[0] * s], [p[2], p[2] + d[2] * s], color=col, lw=1, alpha=0.7)
        tilt = np.degrees(np.arccos(np.clip(-T[2, 2], -1, 1)))
        sx.annotate(f"{nm}: z {p[2] * 100:.1f} cm, {tilt:.0f}° from straight down", (p[0], p[2]),
                    xytext=(8, 8), textcoords="offset points", fontsize=8)
    sx.set_aspect("equal")
    sx.set_xlim(-0.08, 0.66)
    sx.set_ylim(-0.04, 0.42)
    sx.set_xlabel("x (m), forward from the base")
    sx.set_ylabel("z (m)")
    sx.set_title("Side view (to scale)")
    sx.grid(alpha=0.3)
    sx.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("SO-101 plate cell = its sim twin (config/scenes/so101_plate.yaml)", fontsize=12)
    fig.tight_layout()
    out = REPO / "data" / "twin" / "map.png"
    fig.savefig(out, dpi=110)
    print("wrote", out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
