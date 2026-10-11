"""The REAL tag layout: the fixed camera measures where the tags lie relative to each other, a tape measure says
where a few of them are from the robot base, and the two together give every tag's position + turn on the board.

    python3 /repo/tools/tag_layout_from_camera.py                        the camera's view of the tags (relative)
    python3 /repo/tools/tag_layout_from_camera.py --measured "3:0.5,13.2 4:0.2,-14.0 9:30.5,0.4" [--write]
        --measured  tape measurements, id:x,y in CM: x forward of the pan axis, y left of the robot's centre line
        --write     -> config/calibration/<cell>/real_tag_layout.yaml (calibrate.py locate + wrist_tag_check use it)

Why: the tags were placed by hand, cm off the plan. One image fixes their RELATIVE layout well -- each tag's centre:
its ray onto the board plane (the plane from all tags' own poses), averaged over frames; the scale from the tags'
printed size (checked against the depth sensor: 59.3 vs 60 mm, within its 1-2 %). What the image can't know is where
the robot base is: that's the tape measurements. 2 measured tags -> shift + turn; 3 or more -> also the scale, and a
residual per measured tag (a typo or a slipped tape shows up as one tag far off). Measure the tags closest to the base
(3, 4) and one far one (9) carefully: the far one fixes the turn.
No motion.
"""

import argparse
import datetime
import sys
import time

import numpy as np
import rclpy
import yaml
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
from pnp_core.calibration import detect_tags, image_to_gray  # noqa: E402
from pnp_core.catalog import calibration_dir, calibration_targets_path, load_yaml, repo_root  # noqa: E402


def measure(frames=15):
    """-> ({id: (u, v) board-plane centre, m}, {id: in-plane angle of the tag's x axis, rad}, plane basis)."""
    a = load_yaml(calibration_targets_path())["apriltags"]
    want = {int(t["id"]) for t in a["tags"]}
    node = rclpy.create_node("tag_layout_from_camera")
    got = {}
    node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m),
                             qos_profile_sensor_data)
    node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m),
                             qos_profile_sensor_data)
    per = []
    for _ in range(frames):
        got.pop("img", None)
        end = time.time() + 5
        while any(k not in got for k in ("img", "info")) and time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
        if "img" not in got:
            break
        K = np.array(got["info"].k).reshape(3, 3)
        D = np.array(got["info"].d, float) if len(got["info"].d) else np.zeros(5)
        tags = {k: v for k, v in detect_tags(image_to_gray(got["img"]), a["dictionary"], a["size_m"], K, D,
                                             with_corners=True).items() if k in want}
        if len(tags) >= 3:
            per.append((K, tags))
    if not per:
        raise SystemExit("fewer than 3 tags in the fixed camera's view")
    # the board plane in the camera frame (median over all frames / tags): n into the board, d its distance
    n = np.median([T[:3, 2] for _, tags in per for T, _ in tags.values()], axis=0)
    n /= np.linalg.norm(n)
    d = float(np.median([n @ T[:3, 3] for _, tags in per for T, _ in tags.values()]))
    up = -n                                                  # board frame z (up), x / y in the plane: right-handed
    e1 = np.cross([0.0, 1.0, 0.0], up)                       # any in-plane direction
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    cen, ang = {}, {}
    for K, tags in per:
        Ki = np.linalg.inv(K)
        for tid, (T, c) in tags.items():
            ray = Ki @ np.r_[c.mean(axis=0), 1.0]
            p = ray * d / (n @ ray)
            cen.setdefault(tid, []).append([p @ e1, p @ e2])
            x = T[:3, 0]                                     # the tag's x axis (along its top edge, to the right)
            ang.setdefault(tid, []).append(np.arctan2(x @ e2, x @ e1))
    node.destroy_node()
    C = {i: np.median(v, axis=0) for i, v in cen.items()}
    Y = {i: float(np.arctan2(np.median(np.sin(v)), np.median(np.cos(v)))) for i, v in ang.items()}
    spread = max(float(np.max(np.linalg.norm(np.array(v) - C[i], axis=1))) for i, v in cen.items()) * 1000
    print(f"camera: {len(per)} frames, {len(C)} tags, board {d * 100:.1f} cm from the camera along its normal; "
          f"centres repeat within {spread:.1f} mm")
    return C, Y


def fit(C, measured):
    """Measured (base frame, m) <- camera plane: rigid (2 anchors) or similarity (>= 3) -> (s, R, t)."""
    ids = [i for i in measured if i in C]
    A = np.array([C[i] for i in ids])
    B = np.array([measured[i] for i in ids])
    ma, mb = A.mean(axis=0), B.mean(axis=0)
    U, S, Vt = np.linalg.svd((A - ma).T @ (B - mb))
    Dg = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ Dg @ Vt).T
    s = float(np.trace(np.diag(S) @ Dg) / np.sum((A - ma) ** 2)) if len(ids) >= 3 else 1.0
    return s, R, mb - s * R @ ma


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measured", default="", help='"id:x,y id:x,y ..." in cm from the base (x forward, y left)')
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    rclpy.init()
    C, Y = measure()
    plan = {int(t["id"]): t["sim_xy"] for t in load_yaml(calibration_targets_path())["apriltags"]["tags"]}
    ids = sorted(C)
    print("tag-to-tag distances (camera) vs the layout sheet:")
    for i, j in [(3, 4), (7, 8), (5, 6), (1, 2), (3, 7), (4, 8), (3, 9), (4, 9)]:
        if i in C and j in C:
            print(f"  {i}-{j}: {np.linalg.norm(C[i] - C[j]) * 100:5.1f} cm (sheet {np.linalg.norm(np.subtract(plan[i], plan[j])) * 100:5.1f})")
    measured = {}
    for tok in a.measured.split():
        tid, xy = tok.split(":")
        x, y = (float(v) / 100 for v in xy.split(","))
        measured[int(tid)] = np.array([x, y])
    if len(measured) < 2:
        print('\nnext: measure >= 2 tags (best 3, 4 and 9) with a tape and pass --measured "3:x,y 4:x,y 9:x,y" (cm)')
        return 0
    missing = [i for i in measured if i not in C]
    if missing:
        print(f"measured tags not seen by the camera: {missing}")
    s, R, t = fit(C, measured)
    turn = float(np.arctan2(R[1, 0], R[0, 0]))
    print(f"\nfit: {len([i for i in measured if i in C])} measured tags, scale {s:.4f}"
          f"{' (fixed: 2 tags)' if len(measured) < 3 else ''}")
    base = {i: s * R @ C[i] + t for i in ids}
    worst = 0.0
    for i in sorted(measured):
        if i in base:
            r = (base[i] - measured[i]) * 1000
            worst = max(worst, float(np.hypot(*r)))
            print(f"  measured tag {i}: the camera's layout puts it {r[0]:+5.1f} / {r[1]:+5.1f} mm from your measurement")
    print("every tag (x forward, y left of the base, cm; yaw deg; vs the sheet):")
    out = {}
    for i in ids:
        yaw = float(np.degrees(Y[i] + turn))
        yaw = (yaw + 180) % 360 - 180
        out[i] = {"xy": [round(float(base[i][0]), 4), round(float(base[i][1]), 4)], "yaw_deg": round(yaw, 2)}
        print(f"  tag {i}: x {base[i][0] * 100:6.1f}  y {base[i][1] * 100:+6.1f}  yaw {yaw:+5.1f}   "
              f"(sheet {plan[i][0] * 100:5.1f}, {plan[i][1] * 100:+5.1f})")
    if len(measured) >= 3 and worst > 6:
        print(f"WARNING: a measured tag is {worst:.0f} mm off the others' fit: re-measure it (or leave it out)")
    if abs(s - 1) > 0.03:
        print(f"WARNING: scale {s:.3f}: the tape and the camera disagree by > 3 %: check the measurements' units / origin")
    if not a.write:
        print("dry run: add --write to make this the real layout")
        return 0
    path = calibration_dir() / "real_tag_layout.yaml"
    path.write_text("# Where the AprilTags REALLY lie on the board (tools/tag_layout_from_camera.py): their relative layout\n"
                    "# from the fixed camera, anchored to the base by tape measurements. calibrate.py locate and\n"
                    "# tools/wrist_tag_check.py use these instead of the plan (config/calibration_targets_*.yaml).\n"
                    + yaml.safe_dump({"date": datetime.datetime.now().isoformat(timespec="seconds"),
                                      "measured_cm": {int(i): [round(float(v) * 100, 2) for v in m] for i, m in measured.items()},
                                      "scale": round(s, 5), "anchor_residual_mm": round(worst, 2),
                                      "tags": {int(i): v for i, v in out.items()}},
                                     sort_keys=False, default_flow_style=None))
    print(f"written: {path.relative_to(repo_root())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
