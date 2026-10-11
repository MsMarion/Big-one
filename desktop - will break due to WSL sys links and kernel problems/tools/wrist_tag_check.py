"""One-pose wrist camera check against the plate AprilTags (startup, after a bump): is the wrist calibration (and the
arm under it) still right? (tools/wrist_check.py is a different check: wrist depth vs TF vs the sim truth.)

    python3 /repo/tools/wrist_tag_check.py [--save] [--warn-mm 3.5] [--fail-mm 6]

No board to lay out: the plate's AprilTags are the target, where the LAYOUT says they lie (calibration_targets
apriltags: sim_xy / yaw on the board; the frame the fixed camera was located in, calibrate.py locate). Not the drift
monitor's reference: those are single-tag poses from the fixed camera, biased several mm along ITS line of sight
(fine for watching itself, 7 mm off when seen from the wrist, 2026-10-11). The arm goes to one pose from
which the wrist camera sees some of them (searched once in joint space; --save keeps it in
config/calibration/<cell>/wrist_check_pose.yaml for every later check), the tags' reference corners are projected
through the arm + wrist calibration into the wrist image, and the pixel distance to where they ARE seen, as mm at the
tags, says how far off the whole chain is. Not a calibration (one pose can't tell the camera mount from the arm): if
it fails, run the arm accuracy check + tools/fit_joint_offsets.py --with-hand-eye. The result goes to the trial log
(events: wrist_tag_check).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml

sys.path.insert(0, "/repo/tools")
sys.path.insert(0, "/repo")
from calibrate import Calibrator  # noqa: E402
from pnp_core import trials  # noqa: E402
from pnp_core.calibration import (detect_tags, flat_tag_T, image_to_gray, project, tag_corners,  # noqa: E402
                                  tag_layout)
from pnp_core.catalog import calibration_dir, calibration_targets_path, load_yaml  # noqa: E402
from pnp_core.geometry import make_transform  # noqa: E402


def reference_tags(c):
    """{id: T_base_tag} from the layout: flat on the board, printed side up (calibrate.py locate's convention)."""
    from pnp_core.catalog import load_scene
    path = calibration_targets_path()
    t = load_scene()["table"]
    z = t["center"][2] + t["size"][2] / 2 + 0.0008
    # the SURVEYED positions where there are any (calibrate.py locate --survey), else the plan
    sp = calibration_dir() / f"{c.active_set()}_tag_layout.yaml"
    survey = load_yaml(sp) if sp.exists() else None
    out = {tid: flat_tag_T((x, y), yaw, z) for tid, (x, y, yaw) in tag_layout(load_yaml(path)["apriltags"], survey).items()}
    return out or None, Path(path)


def search_pose(c, ref, size, K, wh, T_tcp_opt, n=6000, seed=0):
    """A joint pose from which the wrist camera sees the most reference tags whole, 0.12-0.40 m away, near home."""
    rng = np.random.default_rng(seed)
    lo, hi = np.asarray(c.R.chain.lower), np.asarray(c.R.chain.upper)
    home = np.asarray(c.robot_cfg["arm"]["home"], float)
    corners = tag_corners(size)
    best = None
    for _ in range(n):
        q = rng.uniform(lo, hi)
        T = c.R.chain.fk(q)
        if T[2, 3] < 0.06:                                    # the hand stays well above the board
            continue
        T_opt = T @ T_tcp_opt
        seen = 0
        for T_tag in ref.values():
            P = corners @ T_tag[:3, :3].T + T_tag[:3, 3]
            p = (P - T_opt[:3, 3]) @ T_opt[:3, :3]
            if np.any(p[:, 2] < 0.12) or np.any(p[:, 2] > 0.40):
                continue
            uv = p[:, :2] / p[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
            if np.all((uv >= 12) & (uv <= np.array(wh) - 12)) and np.ptp(uv[:, 0]) >= 25:
                seen += 1
        if seen and (best is None or (seen, -np.abs(q - home).sum()) > (best[0], -np.abs(best[1] - home).sum())):
            best = (seen, q)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", action="store_true", help="keep the chosen pose for later checks")
    ap.add_argument("--warn-mm", type=float, default=3.5)   # sim, servo errors on: 2.6-3.1 mm measured normal
    ap.add_argument("--fail-mm", type=float, default=6.0)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()
    rclpy.init()
    c = Calibrator("wrist", argparse.Namespace(real=a.real, seed=0, settle=1.0, poses=1, dry_run=True, force=False, survey=False, no_move=False,
                                               resolve=None, board=None))
    ref, refpath = reference_tags(c)
    if not ref:
        print(f"no AprilTags in {refpath.name}")
        return 2
    tags = load_yaml(calibration_targets_path())["apriltags"]
    optical = c.s["cam"]["frames"]["color_optical"]
    _, info = c.fresh_frame()
    K = np.array(info.k).reshape(3, 3)
    D = np.array(info.d, float) if len(info.d) else np.zeros(5)
    pose_file = calibration_dir() / "wrist_check_pose.yaml"
    if pose_file.exists():
        q = np.array(yaml.safe_load(pose_file.read_text())["q"], float)
        print(f"check pose: {pose_file.name}")
    else:
        T_tcp_opt = c.tf(c.robot_cfg["frames"]["tcp"], optical)
        found = search_pose(c, ref, tags["size_m"], K, (info.width, info.height), T_tcp_opt)
        if found is None:
            print("no arm pose found from which the wrist camera sees a reference tag whole")
            return 2
        q = found[1]
        print(f"check pose found: {found[0]} tag(s) in view, q = {np.round(q, 3).tolist()}")
        if a.save:
            pose_file.write_text(yaml.safe_dump({"q": [round(float(v), 4) for v in q],
                                                 "note": "tools/wrist_tag_check.py: the wrist camera sees plate tags"}))
            print(f"saved: {pose_file.relative_to(Path('/repo'))}")
    home = c.robot_cfg["arm"]["home"]
    try:
        if c.R.move_joints(q) != 1:
            print("the check pose is not reachable now (an obstacle?)")
            return 2
        time.sleep(1.0)
        img, info = c.fresh_frame()
        T_base_opt = c.tf(c.robot_cfg["frames"]["base"], optical, img.header.stamp)
        seen = detect_tags(image_to_gray(img), tags["dictionary"], tags["size_m"], K, D, with_corners=True)
        rows = []
        for tid, T_ref in ref.items():
            if tid not in seen:
                continue
            T_cam_tag = np.linalg.inv(T_base_opt) @ T_ref
            px = float(np.mean(np.linalg.norm(project(T_cam_tag, tag_corners(tags["size_m"]), K, D) - seen[tid][1],
                                              axis=1)))
            rows.append((tid, px, px * T_cam_tag[2, 3] / K[0, 0] * 1000))
    finally:
        c.R.move_joints(home)
    if not rows:
        print("no reference tag seen from the check pose (covered? moved?)")
        trials.event("wrist_tag_check", "wrist_tag_check", {"state": "NO_TAG"}, "warn")
        return 1
    mm = max(r[2] for r in rows)
    state = "OK" if mm <= a.warn_mm else "WARN" if mm <= a.fail_mm else "FAIL"
    for tid, px, m in rows:
        print(f"  tag {tid}: {px:.2f} px = {m:.2f} mm from where the arm + wrist calibration expect it")
    print(f"wrist tag check: {state} (worst {mm:.2f} mm; WARN > {a.warn_mm}, FAIL > {a.fail_mm})"
          + ("" if state == "OK" else " -> arm accuracy check + tools/fit_joint_offsets.py --with-hand-eye"))
    trials.event("wrist_tag_check", "wrist_tag_check", {"state": state, "worst_mm": round(mm, 2),
                                                "tags": {str(t): round(m, 2) for t, _, m in rows}},
                 "info" if state == "OK" else "warn")
    return 0 if state == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
