"""Is the arm where its encoders say it is? Stationary poses, measured from OUTSIDE the arm's own sensors.

    python3 /repo/tools/arm_accuracy_check.py              (sim: the table board is shown; ~2 min)
    python3 /repo/tools/arm_accuracy_check.py --real       (lay the table board at its spot when asked)
        --poses 6        how many stationary arm poses      --delta-deg 4   approach each from +- this (joint space)

motion_check.py compares where the arm was SENT with where its encoders SAY it is: both come from the encoders, so
a calibration offset or a link bending under gravity is invisible there. This check closes the loop through the
world instead. The table board lies still on the desk; the wrist camera looks at it from several arm poses, and
each view puts the board in the base frame through the whole chain:

    base --(encoders + URDF)--> gripper_link --(wrist calibration)--> D405 --(ChArUco)--> board

If every link of that chain is right, every view puts the board at the SAME place. So:
  spread           how much the board "moves" between poses        = arm kinematics + wrist calibration error
  approach         each pose is reached from both sides (+-delta)  = the servos' dead zone / backlash
  fixed camera     the D435i's own view of the board vs the arm's   = the two calibrations agree (base -> D435i)
  truth (sim)      vs where the sim really put the board
In the sim, run it with and without servo errors (bringup servo_error:=typical, config/servo_error.yaml): the
spread is what joint offsets and gravity flex do at the tool. Every sample is saved (joint angles + board pose) for
fitting a real arm's errors later. docs/learning/17_so101.md "Is the arm where it thinks it is?"
"""

import argparse
import datetime
import json
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

sys.path.insert(0, "/repo/tools")
from calibrate import Calibrator  # noqa: E402

from pnp_core.calibration import board_center_T, detect_board, image_to_gray, mean_transform, pose_error  # noqa: E402
from pnp_core.catalog import load_scene, load_yaml, repo_root  # noqa: E402


def fixed_camera_board(c, b):
    """The board's center in the base frame as the FIXED camera sees it, or None (not the configured position:
    Calibrator.find_table_board falls back to that, which would make this comparison meaningless)."""
    topics = load_yaml("config/interfaces.yaml")["topics"]
    got = {}
    subs = [c.node.create_subscription(Image, topics["color_image"]["name"], lambda m: got.__setitem__("img", m),
                                       qos_profile_sensor_data),
            c.node.create_subscription(CameraInfo, topics["color_info"]["name"], lambda m: got.__setitem__("info", m),
                                       qos_profile_sensor_data)]
    end = time.time() + 4.0
    while ("img" not in got or "info" not in got) and time.time() < end:
        time.sleep(0.05)
    for sub_ in subs:
        c.node.destroy_subscription(sub_)
    if "img" not in got or "info" not in got:
        return None
    T_cb, n, err = detect_board(image_to_gray(got["img"]), b, np.array(got["info"].k).reshape(3, 3), got["info"].d)
    if T_cb is None:
        return None
    return c.tf(c.robot_cfg["frames"]["base"], got["img"].header.frame_id, got["img"].header.stamp) @ T_cb @         board_center_T(b)


def sim_call(c, cmd, **kw) -> dict:
    """A sim command that may time out (the sim answers through the bridge; seen to stall now and then after a long
    run) without losing the measurement: -> {} then."""
    try:
        return c.bench.sim.call(cmd, **kw)
    except RuntimeError as e:
        print(f"  (sim: {e})", flush=True)
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", type=int, default=6)
    ap.add_argument("--delta-deg", type=float, default=4.0)
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--board", choices=["table_board", "hand_board"], default="table_board",
                    help="the PATTERN lying on the desk (calibrate.py --board)")
    ap.add_argument("--max-spread-mm", type=float, default=1.5)
    ap.add_argument("--max-approach-mm", type=float, default=1.0)
    ap.add_argument("--max-fixed-mm", type=float, default=3.0)
    a = ap.parse_args()
    rclpy.init()
    c = Calibrator("wrist", argparse.Namespace(real=a.real, seed=a.seed, settle=a.settle, poses=a.poses,
                                               dry_run=True, force=False, resolve=None, board=a.board))
    servo = {}
    c.node.create_subscription(String, "/sim/servo_error", lambda m: servo.update(json.loads(m.data)),
                               QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                          reliability=ReliabilityPolicy.RELIABLE))
    b, base, cfg = c.s["board"], c.robot_cfg["frames"]["base"], c.robot_cfg
    optical = c.s["cam"]["frames"]["color_optical"]
    home = cfg["arm"]["home"]
    rows, report = [], {"date": datetime.datetime.now().isoformat(timespec="seconds"), "robot": cfg["name"],
                        "board": a.board}
    wrist_truth = None

    def check(name, ok, detail):
        rows.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    try:
        c.R.move_joints(home)
        c.R.gripper(cfg["gripper"]["max_width_m"], 5.0)
        if c.bench:
            c.bench.sim.call("target_away")
            c.bench.sim.call("clear_obstacles")
            c.bench.sim.call("calibration_targets", table_board=True)
        else:
            sp = b["sim_pose"]
            input(f"REAL: clear the work area, lay the table board flat at x={sp['xy'][0]} y={sp['xy'][1]}, long side "
                  f"along x; it must NOT move during the check. Enter ")
        time.sleep(1.0)
        _, info = c.fresh_frame()
        K, size = np.array(info.k).reshape(3, 3), (info.width, info.height)
        D = np.array(info.d, float) if len(info.d) else np.zeros(5)
        cands = c.candidates(K, size, a.poses * 3)
        if not cands or cands[0][0] != "q":
            print("this check samples joint-space views (poses.mode joint in the targets file)")
            return 2
        qs = [q for _, q in cands][:: max(1, len(cands) // a.poses)][: a.poses]
        delta = np.radians(a.delta_deg)
        samples = []
        for i, q in enumerate(qs):
            for side in (+1, -1):
                lo, hi = c.R.chain.lower, c.R.chain.upper
                pre = np.clip(np.asarray(q) + side * delta, lo, hi)
                if c.R.move_joints(pre) != 1 or c.R.move_joints(q) != 1:
                    print(f"  pose {i} ({'+' if side > 0 else '-'}): not reachable, skipped", flush=True)
                    continue
                time.sleep(a.settle)
                img, info = c.fresh_frame()
                T_cb, n, err = detect_board(image_to_gray(img), b, K, D)
                if T_cb is None:
                    print(f"  pose {i} ({'+' if side > 0 else '-'}): board not found", flush=True)
                    continue
                T = c.tf(base, optical, img.header.stamp) @ T_cb @ board_center_T(b)
                js = {j: float(c.R.js[j]) for j in cfg["arm"]["joints"]}
                samples.append({"pose": i, "side": side, "q_cmd": [float(v) for v in q], "q_reported": js,
                                "T_base_board": T.round(6).tolist(), "T_cam_board": T_cb.round(6).tolist(),
                                "corners": n, "reproj_px": round(err, 3)})
                print(f"  pose {i} ({'+' if side > 0 else '-'}): board at {np.round(T[:3, 3] * 1000, 2).tolist()} mm "
                      f"({n} corners, {err:.2f} px)", flush=True)
        T_fixed = fixed_camera_board(c, b) if samples else None
        T_hand_cam = c.tf(c.s["parent"], optical)                  # the wrist calibration in use (for the offset fit)
        ct = sim_call(c, "camera_truth") if c.bench else {}
        truth = ct.get("targets", {}).get("table_board")
        wrist_truth = ct.get("wrist_in_parent")             # sim: the wrist camera's TRUE mount (fit_joint_offsets)
    finally:
        c.R.move_joints(home)
        if c.bench:
            sim_call(c, "calibration_targets", table_board=False)
            sim_call(c, "reset_target")
    if len(samples) < 4:
        print(f"only {len(samples)} views: not enough to judge")
        return 2
    Ts = [np.array(s["T_base_board"]) for s in samples]
    M = mean_transform(Ts)
    dev = np.array([pose_error(T, M) for T in Ts])                       # (mm, deg) of each view from the mean
    print(f"\n{len(samples)} views of the board from {len(qs)} arm poses:")
    check("spread (arm + wrist calibration)", dev[:, 0].max() <= a.max_spread_mm * 2 and
          np.sqrt(np.mean(dev[:, 0] ** 2)) <= a.max_spread_mm,
          f"rms {np.sqrt(np.mean(dev[:, 0] ** 2)):.2f} mm, max {dev[:, 0].max():.2f} mm, rotation rms "
          f"{np.sqrt(np.mean(dev[:, 1] ** 2)):.3f} deg")
    pairs = [(s, t) for s in samples for t in samples if s["pose"] == t["pose"] and s["side"] > t["side"]]
    if pairs:
        d = [pose_error(np.array(s["T_base_board"]), np.array(t["T_base_board"]))[0] for s, t in pairs]
        check("approach direction (dead zone / backlash)", max(d) <= a.max_approach_mm,
              f"same pose from + vs - : mean {np.mean(d):.2f} mm, max {max(d):.2f} mm")
    if T_fixed is not None:
        e = pose_error(M, T_fixed)
        check("fixed camera agrees", e[0] <= a.max_fixed_mm, f"the D435i puts the board {e[0]:.2f} mm / {e[1]:.2f} deg "
              f"from where the arm + wrist camera put it")
    else:
        print("  (fixed camera: the D435i doesn't resolve the board's markers from where it is: not compared)")
    sp = b["sim_pose"]
    e = float(np.hypot(M[0, 3] - sp["xy"][0], M[1, 3] - sp["xy"][1])) * 1000
    print(f"  configured spot: the board's center is {e:.1f} mm from x={sp['xy'][0]} y={sp['xy'][1]} "
          f"(only as good as you laid it there)")
    if truth is not None:
        e = pose_error(M, np.array(truth) @ board_center_T(b))           # the sim's plate frame = the board's
        print(f"  truth (sim): the arm + wrist chain puts the board {e[0]:.2f} mm / {e[1]:.3f} deg from where it is")
        report["truth_error"] = {"pos_mm": round(e[0], 3), "rot_deg": round(e[1], 4)}
    if servo:
        print(f"  servo errors in the sim ({servo.get('profile')}): " + ", ".join(
            f"{j} {v['offset_deg']:+.2f} deg offset" for j, v in servo.get("joints", {}).items()))
        report["servo_error"] = servo
    sp_T = np.eye(4)
    sp_T[:3, :3] = np.array([[np.cos(np.radians(sp["yaw_deg"])), np.sin(np.radians(sp["yaw_deg"])), 0],
                             [np.sin(np.radians(sp["yaw_deg"])), -np.cos(np.radians(sp["yaw_deg"])), 0], [0, 0, -1.0]])
    t = load_scene()["table"]
    sp_T[:3, 3] = [sp["xy"][0], sp["xy"][1], t["center"][2] + t["size"][2] / 2]   # the configured spot, on the desk
    report["references"] = {k: v for k, v in (
        ("truth", None if truth is None else (np.array(truth) @ board_center_T(b)).round(6).tolist()),
        ("fixed_camera", None if T_fixed is None else T_fixed.round(6).tolist()),
        ("configured_spot", sp_T.round(6).tolist())) if v is not None}
    report["T_hand_cam"] = T_hand_cam.round(6).tolist()
    if wrist_truth is not None:
        report["wrist_truth"] = wrist_truth
    report["calibration_set"] = c.active_set()
    report["hand_link"] = c.s["parent"]
    report["source"] = "real" if a.real else f"sim_{servo.get('profile', 'none')}"
    report.update({"samples": samples, "checks": {n: ok for n, ok in rows}})
    out = repo_root() / "data" / "calibration" / f"arm_accuracy_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    n_ok = sum(ok for _, ok in rows)
    print(f"RESULT: {n_ok}/{len(rows)} passed   (samples: {out.relative_to(repo_root())})")
    rclpy.try_shutdown()
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
