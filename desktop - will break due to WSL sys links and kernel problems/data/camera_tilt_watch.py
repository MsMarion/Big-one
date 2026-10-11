"""Live guide for aiming the fixed camera: which tags it sees whole, how close each is to the image edge, and the
camera's tilt + height (from the tags' own poses: no calibration needed). A line when the set of seen tags changes
or the tilt moves by > 1 deg; "ALL 9 IN VIEW" when they are (with the tightest margin to an edge). No motion.
    docker exec pnp_dev bash /repo/data/rospy.sh -u /repo/data/camera_tilt_watch.py
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
from pnp_core.calibration import detect_tags, image_to_gray  # noqa: E402
from pnp_core.catalog import calibration_targets_path, load_yaml  # noqa: E402

A = load_yaml(calibration_targets_path())["apriltags"]
WANT = sorted(int(t["id"]) for t in A["tags"])
rclpy.init()
node = rclpy.create_node("camera_tilt_watch")
got = {}
node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m), qos_profile_sensor_data)
node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m),
                         qos_profile_sensor_data)
last_ids, last_tilt, t_last = None, None, 0.0
STEADY_S = 3.0
cand = [None, 0.0]
while True:
    got.pop("img", None)
    end = time.time() + 3
    while ("img" not in got or "info" not in got) and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if "img" not in got:
        print("no image from the D435i", flush=True)
        continue
    m = got["img"]
    K = np.array(got["info"].k).reshape(3, 3)
    tags = {k: v for k, v in detect_tags(image_to_gray(m), A["dictionary"], A["size_m"], K, None, with_corners=True).items()
            if k in WANT}
    ids = sorted(tags)
    if not tags:
        continue
    tilt = float(np.median([np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))) for T, _ in tags.values()]))
    height = float(np.median([T[:3, 2] @ T[:3, 3] for T, _ in tags.values()]))
    # each tag's closest corner to an image edge; which edge is tight
    margins = {}
    for tid, (_, c) in tags.items():
        e = {"top": c[:, 1].min(), "bottom": m.height - c[:, 1].max(), "left": c[:, 0].min(), "right": m.width - c[:, 0].max()}
        side = min(e, key=e.get)
        margins[tid] = (float(e[side]), side)
    tight = min(margins, key=lambda k: margins[k][0])
    # report only a STEADY view (the same tags for STEADY_S, tilt within 1 deg): no stream while things move
    sig = (tuple(ids), round(tilt))
    if sig != cand[0]:
        cand[:] = [sig, time.time()]
    steady = time.time() - cand[1] >= STEADY_S
    if steady and (ids != last_ids or last_tilt is None or abs(tilt - last_tilt) > 1.0):
        missing = [t for t in WANT if t not in ids]
        head = "ALL 9 IN VIEW" if not missing else f"{len(ids)}/9, missing {missing}"
        print(f"{head} | tilt {tilt:.1f} deg from straight down, {height * 100:.1f} cm up | tightest: tag {tight} "
              f"{margins[tight][0]:.0f} px from the {margins[tight][1]} edge", flush=True)
        last_ids, last_tilt, t_last = ids, tilt, time.time()
    time.sleep(0.4)
