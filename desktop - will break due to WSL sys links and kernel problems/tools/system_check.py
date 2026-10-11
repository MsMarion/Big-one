"""System check: is every topic in the interface contract alive, at the right rate, with the right format?

Run inside the pnp container while the sim (or the real robot) is running:
    python3 /repo/tools/system_check.py --sim            # includes sim-only topics (/clock, ground truth)
    python3 /repo/tools/system_check.py --seconds 5

Reads config/interfaces.yaml (the contract) and config/camera_d435i.yaml. Exit code 1 if anything fails.
What this teaches: write the contract down once, then *test* it - most integration bugs are a wrong
topic name, frame id, encoding or rate, and they are cheap to catch here and expensive to catch later.
"""

import argparse
import sys
import time
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnp_core.catalog import load_fixed_camera  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="also check sim-only topics")
    ap.add_argument("--stack", action="store_true", help="also check topics produced by the ROS stack (bringup.launch.py)")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    contract = yaml.safe_load((REPO / "config/interfaces.yaml").read_text())
    cam = load_fixed_camera()
    topics = {k: v for k, v in contract["topics"].items()
              if v.get("min_hz", 0) > 0 and (args.sim or not v.get("sim_only"))
              and (args.stack or not v.get("stack"))}

    rclpy.init()
    node = Node("pnp_system_check")
    stats = {k: {"n": 0, "last": None} for k in topics}

    def make_cb(key):
        def cb(msg):
            stats[key]["n"] += 1
            stats[key]["last"] = msg
        return cb

    for key, t in topics.items():
        node.create_subscription(get_message(t["type"]), t["name"], make_cb(key), qos_profile_sensor_data)

    gt_frame = contract["frames"]["object_ground_truth"]
    gt = {"n": 0, "parent": None, "xyz": None}

    def tf_cb(msg):
        for tr in msg.transforms:
            if tr.child_frame_id == gt_frame:
                gt["n"] += 1
                gt["parent"] = tr.header.frame_id
                v = tr.transform.translation
                gt["xyz"] = (round(v.x, 3), round(v.y, 3), round(v.z, 3))

    if args.sim:
        node.create_subscription(get_message("tf2_msgs/msg/TFMessage"), "/tf", tf_cb, 100)

    node.get_logger().info(f"listening to {len(topics)} topics for {args.seconds:.0f} s ...")
    # wait for discovery, then measure
    t_end = time.time() + 2.0
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.1)
    for s in stats.values():
        s["n"] = 0
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    dt = time.time() - t0

    ok = True
    print(f"\n{'topic':55s} {'rate':>8s} {'min':>6s}  details")
    for key, t in topics.items():
        hz = stats[key]["n"] / dt
        msg = stats[key]["last"]
        detail, good = "", hz >= t["min_hz"]
        if msg is not None and hasattr(msg, "encoding"):
            detail = f"{msg.width}x{msg.height} {msg.encoding} frame={msg.header.frame_id}"
            good &= (msg.width, msg.height) == (cam["width"], cam["height"])
            good &= msg.header.frame_id == cam["frames"]["color_optical"]
        elif msg is not None and hasattr(msg, "k"):
            detail = f"fx={msg.k[0]:.1f} fy={msg.k[4]:.1f} cx={msg.k[2]:.1f} cy={msg.k[5]:.1f}"
            good &= abs(msg.k[0] - cam["fx"]) < 1.0 and abs(msg.k[4] - cam["fy"]) < 1.0
        elif msg is not None and hasattr(msg, "name") and hasattr(msg, "position"):
            detail = f"{len(msg.name)} joints"
        ok &= good
        print(f"{t['name']:55s} {hz:8.1f} {t['min_hz']:6.0f}  {'OK  ' if good else 'FAIL'} {detail}")

    # Duplicate node names = leftover processes from an earlier run (e.g. two detectors answering the same
    # service with different results, or two controller managers fighting). Always a bug.
    names = [f"{ns.rstrip('/')}/{n}" for n, ns in node.get_node_names_and_namespaces()]
    dups = sorted({x for x in names if names.count(x) > 1 and "_NODE_NAME_UNKNOWN_" not in x
                   and "transform_listener" not in x})
    ok &= not dups
    print(f"{'duplicate ROS nodes':55s} {len(dups):8d} {'':>6s}  {'OK  ' if not dups else 'FAIL'} {', '.join(dups)}")

    if args.sim:
        good = gt["n"] > 0 and gt["parent"] == contract["frames"]["base"]
        ok &= good
        print(f"{'TF ' + str(gt['parent']) + ' -> ' + gt_frame + ' (ground truth)':55s} {gt['n'] / dt:8.1f} {'':>6s}  "
              f"{'OK  ' if good else 'FAIL'} xyz={gt['xyz']}")

    node.destroy_node()
    rclpy.shutdown()
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
