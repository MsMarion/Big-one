"""How well does the 6-D pose node track the can, before and DURING a pick? (sim, ground truth from Isaac)

    python3 /repo/tools/pose_track_check.py [--seconds 10] [--pick]     (stack with pose_estimation:=true)

Samples /perception/object_pose and /perception/pose_status, compares each pose with the TF `target` (Isaac's
ground truth at the pose's stamp time), and prints a timeline (one line per 0.5 s): state, position error [mm],
orientation error [deg] (axis tilt for a can; full rotation modulo its symmetry for a box), track time.
--pick runs one pick & place meanwhile, so you see what happens while the gripper covers the object (HOLD),
carries it (ATTACHED), and puts it down.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.time import Time
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_benchmark import Bench  # noqa: E402

from pnp_core.geometry import quat_to_matrix  # noqa: E402
from pnp_core.object_grasp import up_axis  # noqa: E402

from pnp_core.catalog import load_robot as _load_robot  # noqa: E402
BASE = _load_robot()["frames"]["base"]     # the active cell's robot base frame


def rot_about(axis, angle):
    """Rotation by `angle` about a unit axis (Rodrigues)."""
    k = np.asarray(axis, float)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--pick", action="store_true")
    a = ap.parse_args()
    rclpy.init()
    b = Bench(name="pnp_pose_track_check")
    samples, status, stats = [], {}, {"no_gt_at_stamp": 0}
    _, obj = b.scene()
    sym = None                                    # rotations about the up axis that leave the object unchanged
    if not obj["grasp"].get("symmetric_yaw"):
        n = int(round(360 / obj["grasp"].get("yaw_period_deg", 360)))
        sym = [rot_about(up_axis(obj), 2 * np.pi * k / n) for k in range(n)]

    def on_pose(m: PoseStamped):
        # ground truth AT THE POSE'S TIME: while the can moves (carried), comparing with the newest truth instead
        # would count the age of the pose as error (0.5 m/s x 20 ms = 10 mm)
        try:
            t = b.tf.lookup_transform(BASE, "target", Time.from_msg(m.header.stamp)).transform
        except Exception:  # noqa: BLE001
            stats["no_gt_at_stamp"] += 1
            return
        p = m.pose
        R = quat_to_matrix([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])
        Rg = quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])
        pos_err = np.linalg.norm([p.position.x - t.translation.x, p.position.y - t.translation.y,
                                  p.position.z - t.translation.z]) * 1000
        if sym is None:                                            # rotationally symmetric: only the axis counts
            axis_err = np.degrees(np.arccos(np.clip(abs(R[:, 1] @ Rg[:, 1]), -1, 1)))   # mesh y = the up axis
        else:                                                      # full rotation error, modulo the yaw symmetry
            axis_err = min(np.degrees(np.arccos(np.clip((np.trace(R.T @ Rg @ S) - 1) / 2, -1, 1))) for S in sym)
        samples.append((time.time(), status.get("state", "?"), pos_err, axis_err, status.get("last_track_ms")))

    b.create_subscription(PoseStamped, "/perception/object_pose", on_pose, 10)
    b.create_subscription(String, "/perception/pose_status", lambda m: status.update(json.loads(m.data)), 10)
    t0 = time.time()
    b.spin_for(1.0)
    if a.pick:
        st = b.sim.call("state")
        print(f"pick & place -> goal {np.round(st['goal']['xy'], 3).tolist()} while tracking...", flush=True)
        res = b.pick_place(st["goal"]["xy"], b.scene()[1]["yolo_class"], "dev")
        print(f"pick & place: {res.outcome_name if res else 'no result'}", flush=True)
    b.spin_for(max(0.0, a.seconds - (time.time() - t0)))
    if not samples:
        print("no /perception/object_pose received (pose_estimation:=true? image pnp:pose?)")
        return 1
    print(f"{'t [s]':>6s} {'state':10s} {'pos err [mm]':>12s} {'rot err [deg]':>14s} {'track [ms]':>10s}")
    nxt = samples[0][0]
    for t, s, pe, ae, tr in samples:
        if t >= nxt:
            print(f"{t - t0:6.1f} {s:10s} {pe:12.1f} {ae:14.1f} {str(tr):>10s}")
            nxt = t + 0.5
    for state in ("TRACKING", "ATTACHED"):
        pe = np.array([s[2] for s in samples if s[1] == state])
        ae = np.array([s[3] for s in samples if s[1] == state])
        if len(pe):
            print(f"{state} samples: {len(pe)}, position error median {np.median(pe):.1f} mm, max {pe.max():.1f} mm; "
                  f"rotation error median {np.median(ae):.1f} deg, max {ae.max():.1f} deg")
    print(f"registrations {status.get('registrations')}, last register {status.get('last_register_ms')} ms; "
          f"poses without ground truth at their stamp: {stats['no_gt_at_stamp']}")
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
