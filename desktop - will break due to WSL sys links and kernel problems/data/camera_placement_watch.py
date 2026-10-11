"""Live guide while moving the fixed camera's stand: where is the D435i now vs where the tag layout was planned for?

From each seen tag's own pose (single-tag PnP, ~1 cm good: enough to steer by) and its PLANNED spot on the board ->
the camera lens in the base frame. Prints a line when the estimate moves > 1 cm or the set of seen tags changes,
and "ON TARGET" when within 1.5 cm. No motion, nothing written.
    docker exec pnp_dev bash /repo/data/rospy.sh -u /repo/data/camera_placement_watch.py
"""
import sys
import time

import numpy as np
import rclpy
import yaml
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
from pnp_core.calibration import detect_tags, image_to_gray  # noqa: E402

TARGET = np.array(yaml.safe_load(open("/repo/config/calibration/so101_plate/real_d435i.yaml"))["T_parent_optical"]["xyz"])
A = yaml.safe_load(open("/repo/config/calibration_targets_so101_plate.yaml"))["apriltags"]
PLAN = {int(t["id"]): t["sim_xy"] for t in A["tags"]}
BOARD_Z = -0.0024

rclpy.init()
node = rclpy.create_node("camera_placement_watch")
got = {}
node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m), qos_profile_sensor_data)
node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m),
                         qos_profile_sensor_data)
last, last_ids, t_last = None, None, 0.0
print(f"target: lens at x {TARGET[0] * 100:.1f} cm forward, y {TARGET[1] * 100:+.1f} cm, "
      f"{(TARGET[2] - BOARD_Z) * 100:.1f} cm above the board", flush=True)
while True:
    got.pop("img", None)
    end = time.time() + 3
    while ("img" not in got or "info" not in got) and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "img" not in got:
        print("no image from the D435i", flush=True)
        continue
    m, info = got["img"], got["info"]
    K, D = np.array(info.k).reshape(3, 3), np.array(info.d, float) if len(info.d) else np.zeros(5)
    tags = detect_tags(image_to_gray(m), A["dictionary"], A["size_m"], K, D)
    est = []
    for tid, T in tags.items():
        if tid not in PLAN:
            continue
        p = -T[:3, :3].T @ T[:3, 3]          # the camera in the tag's frame (x = base x, y = -base y, z into the board)
        est.append([PLAN[tid][0] + p[0], PLAN[tid][1] - p[1], BOARD_Z - p[2]])
    ids = sorted(tags)
    if not est:
        if ids != last_ids:
            print(f"no layout tags in view (seen: {ids})", flush=True)
        last_ids = ids
        continue
    c = np.median(np.array(est), axis=0)
    d = c - TARGET
    moved = last is None or np.linalg.norm(c - last) > 0.01 or ids != last_ids or time.time() - t_last > 20
    if moved:
        on = np.hypot(d[0], d[1]) < 0.015 and abs(d[2]) < 0.02
        hint = []
        if abs(d[0]) >= 0.01:
            hint.append(f"{'BACK (away from the robot)' if d[0] < 0 else 'FORWARD (toward the robot)'} {abs(d[0]) * 100:.0f} cm")
        if abs(d[1]) >= 0.01:
            hint.append(f"to the robot's {'LEFT' if d[1] < 0 else 'RIGHT'} {abs(d[1]) * 100:.0f} cm")
        print(f"tags {ids} ({len(ids)}/9) | lens x {c[0] * 100:.1f} y {c[1] * 100:+.1f} cm, {(c[2] - BOARD_Z) * 100:.1f} cm up | "
              + ("ON TARGET" if on else "move " + ", ".join(hint) if hint else "height off"), flush=True)
        last, last_ids, t_last = c, ids, time.time()
    time.sleep(0.5)
