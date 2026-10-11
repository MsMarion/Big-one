"""How well does the camera measure the scene? Measured boxes vs the sim's true boxes, as a table.

    python3 /repo/tools/scene_check.py [--frames 10]      (stack running with sensor_view:=true)

Averages /scene/model (pnp_perception/scene_model_node) over a few frames and prints, per true box:
measured size vs true size, size error per axis (mm), center error (mm), yaw error (deg).
How to read it (pnp_core/scene_model.py docstring has the details):
  sizes off by the same % in every direction -> depth scale or focal length (intrinsics)
  positions off but sizes right               -> camera pose (extrinsics / hand-eye calibration)
  height too small                            -> the top face isn't fully seen (occlusion, steep view)
"""

import argparse
import json
import sys
import time

import numpy as np
import rclpy
from std_msgs.msg import String


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    a = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("pnp_scene_check")
    frames = []
    n.create_subscription(String, "/scene/model", lambda m: frames.append(json.loads(m.data)), 10)
    end = time.time() + 20
    while len(frames) < a.frames and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    rclpy.try_shutdown()
    if not frames:
        print("no /scene/model messages (is the stack running with sensor_view:=true?)")
        return 1
    last = frames[-1]
    print(f"{len(last['measured'])} object(s) measured in the last frame (averaging {len(frames)} frames)\n")
    per_box = {}
    for f in frames:
        for r in f["vs_truth"]:
            per_box.setdefault(r["id"], []).append(r)
    print(f"{'true box':8s} {'measured size [cm]':22s} {'size error [mm] (L/W/H)':26s} {'center [mm]':>11s} {'yaw [deg]':>9s}")
    for bid, recs in per_box.items():
        ok = [r for r in recs if r["matched"]]
        if not ok:
            print(f"{bid:8s} NOT SEEN (occluded, too far, or merged with another object)")
            continue
        e = np.mean([r["size_err_mm"] for r in ok], axis=0)
        c = np.mean([r["center_err_mm"] for r in ok])
        y = np.mean([abs(r["yaw_err_deg"]) for r in ok])
        m = last["measured"][ok[-1]["measured_index"]]["size"] if ok[-1]["measured_index"] < len(last["measured"]) else [0, 0, 0]
        print(f"{bid:8s} {m[0] * 100:5.1f} x {m[1] * 100:5.1f} x {m[2] * 100:5.1f}     "
              f"{e[0]:+6.1f} / {e[1]:+6.1f} / {e[2]:+6.1f}        {c:8.1f} {y:9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
