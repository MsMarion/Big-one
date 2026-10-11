"""Where the plate's AprilTags lie, measured BY THE ARM: no tape, no manual placement.

    python3 /repo/tools/arm_tag_survey.py --real [--views 2] [--write] [--then-locate]

The fixed camera knows the tags' layout RELATIVE to each other well (tools/tag_layout_from_camera.py: 0.5 mm
repeatable) but not where the robot base is. The arm does: for each tag it can see, the wrist camera (D405) looks at
it from a pose near home, 12-30 cm away, and the tag's centre pixel is projected onto the board plane through the
arm (FK of the MEASURED joints, the joint offsets, the wrist calibration: all in TF at the image's stamp). Each tag
from `--views` different poses (their arm errors differ; the spread is printed). Then the camera's relative layout
is fitted to those points (shift + turn + scale) -> every tag's position on the board, and a residual per tag:
how well the arm + wrist calibration agree with the fixed camera across the board (a check of the arm, too).
--write: config/calibration/<cell>/real_tag_layout.yaml; --then-locate: also calibrate the fixed camera on it
(calibrate.py locate) and record the drift reference.

Safety: the board is added to MoveIt as a collision box first; every move is MoveIt-planned at the real arm's speed
cap (PNP_MAX_VELOCITY_SCALING, 15 %); the hand stays >= 6 cm above the board; home at the end (also on an error).
"""
import argparse
import datetime
import subprocess
import sys
import time

import numpy as np
import rclpy
import yaml

sys.path.insert(0, "/repo/tools")
sys.path.insert(0, "/repo")
from calibrate import Calibrator  # noqa: E402
from tag_layout_from_camera import fit, measure  # noqa: E402
from pnp_core import trials  # noqa: E402
from pnp_core.calibration import detect_tags, image_to_gray, tag_corners  # noqa: E402
from pnp_core.catalog import calibration_dir, calibration_targets_path, load_scene, load_yaml, repo_root  # noqa: E402
from wrist_tag_check import reference_tags  # noqa: E402


# the hand stays within the scene's reach band: at ~28 cm on the 12 V 2 A supply the elbow hit 81 % load and every
# servo dropped off the bus (config/scenes/so101_plate_tags.yaml reach_m [0.13, 0.22])
REACH_M = 0.22


def poses_for(c, T_tag, size, K, wh, T_tcp_opt, n_views, rng, n=5000):
    """Up to n_views joint poses from which the wrist camera sees this tag whole, 12-30 cm away, the tag near the
    image centre, the hand >= 6 cm above the board, near home; different from each other (>= 0.25 rad apart)."""
    lo, hi = np.asarray(c.R.chain.lower), np.asarray(c.R.chain.upper)
    home = np.asarray(c.robot_cfg["arm"]["home"], float)
    P = tag_corners(size) @ T_tag[:3, :3].T + T_tag[:3, 3]
    cands = []
    for _ in range(n):
        q = rng.uniform(lo, hi)
        T = c.R.chain.fk(q)
        if T[2, 3] < 0.06 or np.hypot(T[0, 3], T[1, 3]) > REACH_M:
            continue
        T_opt = T @ T_tcp_opt
        p = (P - T_opt[:3, 3]) @ T_opt[:3, :3]
        if np.any(p[:, 2] < 0.12) or np.any(p[:, 2] > 0.30):
            continue
        uv = p[:, :2] / p[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        if not np.all((uv >= 20) & (uv <= np.array(wh) - 20)) or np.ptp(uv[:, 0]) < 30:
            continue
        off = np.linalg.norm(uv.mean(axis=0) - np.array(wh) / 2) / np.linalg.norm(wh)     # 0 = centred
        cands.append((off + 0.15 * np.abs(q - home).sum(), q))
    cands.sort(key=lambda t: t[0])
    out = []
    for _, q in cands:
        if all(np.max(np.abs(q - o)) > 0.25 for o in out):
            out.append(q)
        if len(out) == n_views:
            break
    return out


def look(c, tid, tags_cfg, K, D, board_z, frames=5):
    """The tag's centre on the board plane, from the wrist camera at the arm's CURRENT pose -> (xy, n frames)."""
    optical = c.s["cam"]["frames"]["color_optical"]
    base = c.robot_cfg["frames"]["base"]
    pts = []
    for _ in range(frames):
        img, _ = c.fresh_frame(skip=2)
        if img is None:
            break
        seen = detect_tags(image_to_gray(img), tags_cfg["dictionary"], tags_cfg["size_m"], K, D, with_corners=True)
        if tid not in seen:
            continue
        T = c.tf(base, optical, img.header.stamp)
        ray = T[:3, :3] @ (np.linalg.inv(K) @ np.r_[seen[tid][1].mean(axis=0), 1.0])
        s = (board_z - T[2, 3]) / ray[2]
        if s > 0:
            pts.append((T[:3, 3] + s * ray)[:2])
    return (np.median(pts, axis=0), len(pts)) if pts else (None, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--views", type=int, default=2)
    ap.add_argument("--tags", default="", help="only these ids (e.g. 3,4,9)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--then-locate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true", help="list each tag's view poses, move nothing")
    a = ap.parse_args()
    rclpy.init()
    print("the fixed camera: the tags' layout relative to each other ...", flush=True)
    C, Y = measure()
    c = Calibrator("wrist", argparse.Namespace(real=a.real, seed=a.seed, settle=1.0, poses=1, dry_run=True,
                                               force=False, survey=False, no_move=False, resolve=None, board=None))
    ref, _ = reference_tags(c)                      # the current layout estimate: where to point the wrist camera
    tags_cfg = load_yaml(calibration_targets_path())["apriltags"]
    t = load_scene()["table"]
    board_z = t["center"][2] + t["size"][2] / 2
    tag_z = board_z + 0.0008
    # the board as an obstacle, 5 mm lower than it is: the base stands flush ON it (touching = "in collision" for
    # MoveIt); the hand keeps >= 6 cm above it anyway. Left in the scene afterwards (harmless, and safer)
    c.R.add_box("survey_board", [t["center"][0], t["center"][1], t["center"][2] - 0.005], t["size"])
    _, info = c.fresh_frame()
    K = np.array(info.k).reshape(3, 3)
    D = np.array(info.d, float) if len(info.d) else np.zeros(5)
    # tcp -> wrist optical from the arm model + the camera TF (NOT tf(tcp, optical): over the Jetson link /tf_static
    # carries only its LAST publisher -- the camera calibration's -- so the URDF's fixed frames, the tcp among them,
    # never reach the PC; base -> optical does: moving joints + the camera's static)
    T_tcp_opt = np.linalg.inv(c.R.chain.fk(c.R.arm_q())) @ c.tf(c.robot_cfg["frames"]["base"],
                                                                c.s["cam"]["frames"]["color_optical"])
    rng = np.random.default_rng(a.seed)
    only = {int(v) for v in a.tags.split(",") if v}
    arm = {}
    home = c.robot_cfg["arm"]["home"]
    try:
        # around the base in one sweep (by bearing), not side to side
        for tid in sorted(ref, key=lambda i: np.arctan2(ref[i][1, 3], ref[i][0, 3])):
            if (only and tid not in only) or tid not in C:
                continue
            qs = poses_for(c, ref[tid], tags_cfg["size_m"], K, (info.width, info.height), T_tcp_opt, a.views, rng)
            if not qs:
                print(f"tag {tid}: no arm pose sees it (out of reach)", flush=True)
                continue
            if a.plan_only:
                for q in qs:
                    T = c.R.chain.fk(q)
                    print(f"tag {tid}: view pose {np.round(np.degrees(q), 1).tolist()} deg, hand at "
                          f"{np.round(T[:3, 3] * 100, 1).tolist()} cm", flush=True)
                continue
            got = []
            for q in qs:
                if c.R.move_joints(q) != 1:
                    print(f"tag {tid}: a view pose wasn't reachable now, skipped", flush=True)
                    continue
                time.sleep(1.2)
                xy, n = look(c, tid, tags_cfg, K, D, tag_z)
                if xy is not None:
                    got.append(xy)
            if got:
                arm[tid] = np.mean(got, axis=0)
                spread = max(np.linalg.norm(g - arm[tid]) for g in got) * 1000
                print(f"tag {tid}: the arm sees it at x {arm[tid][0] * 100:.1f}, y {arm[tid][1] * 100:+.1f} cm "
                      f"({len(got)} views, spread {spread:.1f} mm)", flush=True)
            else:
                print(f"tag {tid}: not detected by the wrist camera from its view poses", flush=True)
    finally:
        if not a.plan_only:
            c.R.move_joints(home)
    if a.plan_only:
        return 0
    if len(arm) < 2:
        print(f"the arm measured {len(arm)} tag(s): need >= 2")
        return 1
    s, R, tr = fit(C, arm)
    turn = float(np.arctan2(R[1, 0], R[0, 0]))
    res = {i: (s * R @ C[i] + tr - arm[i]) * 1000 for i in arm}
    rms = float(np.sqrt(np.mean([np.sum(r ** 2) for r in res.values()])))
    print(f"\nfit: the camera's layout onto {len(arm)} arm-measured tags, scale {s:.4f}, rms {rms:.1f} mm")
    for i in sorted(res):
        print(f"  tag {i}: arm vs camera layout {res[i][0]:+5.1f} / {res[i][1]:+5.1f} mm")
    out = {}
    for i in sorted(C):
        p = s * R @ C[i] + tr
        yaw = (float(np.degrees(Y[i] + turn)) + 180) % 360 - 180
        out[int(i)] = {"xy": [round(float(p[0]), 4), round(float(p[1]), 4)], "yaw_deg": round(yaw, 2)}
        print(f"  tag {i}: x {p[0] * 100:6.1f}  y {p[1] * 100:+6.1f} cm  yaw {yaw:+5.1f}")
    trials.event("arm_tag_survey", "arm_tag_survey", {"tags": len(arm), "rms_mm": round(rms, 2), "scale": round(s, 5),
                                                      "residual_mm": {str(i): round(float(np.hypot(*r)), 2)
                                                                      for i, r in res.items()}})
    if not a.write:
        print("dry run: --write makes this the real layout")
        return 0
    path = calibration_dir() / "real_tag_layout.yaml"
    path.write_text("# Where the AprilTags REALLY lie on the board (tools/arm_tag_survey.py): their relative layout from the\n"
                    "# fixed camera, anchored to the base by the ARM (wrist camera views of the tags). calibrate.py locate\n"
                    "# and tools/wrist_tag_check.py use these instead of the plan.\n"
                    + yaml.safe_dump({"date": datetime.datetime.now().isoformat(timespec="seconds"), "measured_by": "arm",
                                      "arm_tags": {int(i): [round(float(v), 4) for v in p] for i, p in arm.items()},
                                      "scale": round(s, 5), "rms_mm": round(rms, 2), "tags": out},
                                     sort_keys=False, default_flow_style=None))
    print(f"written: {path.relative_to(repo_root())}")
    if a.then_locate:
        for cmd in (["python3", "/repo/tools/calibrate.py", "locate", "--real", "--no-move"],
                    ["python3", "/repo/tools/calibrate.py", "reference", "--real"]):
            print(f"\n$ {' '.join(cmd[1:])}", flush=True)
            subprocess.run(cmd, input="\n", text=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
