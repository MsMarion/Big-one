"""Are the printed tags really 60 mm? The D435i's depth at each tag's centre vs the distance its pose implies when
the tag is ASSUMED 60 mm (a pose's distance scales with the assumed size) -> the printed size. No motion.
    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/tag_size_check.py"""
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
from pnp_core.calibration import detect_tags, image_to_gray  # noqa: E402
from pnp_core.catalog import calibration_targets_path, load_yaml  # noqa: E402
from pnp_core.perception_math import depth_to_meters  # noqa: E402

rclpy.init()
node = rclpy.create_node("tag_size_check")
got = {}
node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m), qos_profile_sensor_data)
node.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw", lambda m: got.__setitem__("depth", m),
                         qos_profile_sensor_data)
node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m), qos_profile_sensor_data)
a = load_yaml(calibration_targets_path())["apriltags"]
ratios = {}
for _ in range(8):                                  # a few frames
    got.pop("img", None), got.pop("depth", None)
    end = time.time() + 5
    while any(k not in got for k in ("img", "depth", "info")) and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    m, d, info = got["img"], got["depth"], got["info"]
    K = np.array(info.k).reshape(3, 3)
    depth = depth_to_meters(np.frombuffer(d.data, np.uint16).reshape(d.height, d.width), d.encoding)
    for tid, (T, c) in detect_tags(image_to_gray(m), a["dictionary"], a["size_m"], K, None, with_corners=True).items():
        u, v = c.mean(axis=0).astype(int)
        patch = depth[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4]
        z = np.median(patch[patch > 0]) if np.any(patch > 0) else np.nan
        if np.isfinite(z):
            ratios.setdefault(tid, []).append(z / T[2, 3])     # depth z vs the pose's z (both along the optical axis)
for tid in sorted(ratios):
    r = np.median(ratios[tid])
    print(f"tag {tid}: depth / pose-distance {r:.3f} -> printed black square ~{a['size_m'] * r * 1000:.1f} mm")
allr = np.median([np.median(v) for v in ratios.values()])
print(f"ALL: printed size ~{a['size_m'] * allr * 1000:.1f} mm (assumed {a['size_m'] * 1000:.0f}); depth itself is good to ~1-2 %")
