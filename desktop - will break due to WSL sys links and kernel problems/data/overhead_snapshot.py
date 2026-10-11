"""One fixed-camera (D435i) frame -> data/overhead_view.png with the AprilTags it finds marked + listed (no motion).
    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/overhead_snapshot.py"""
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
from pnp_core.calibration import detect_tags, image_to_gray  # noqa: E402
from pnp_core.catalog import calibration_targets_path, load_yaml  # noqa: E402

rclpy.init()
node = rclpy.create_node("overhead_snapshot")
got = {}
node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m), qos_profile_sensor_data)
node.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m), qos_profile_sensor_data)
end = time.time() + 10
while ("img" not in got or "info" not in got) and time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.1)
m, info = got["img"], got["info"]
gray = image_to_gray(m)
a = load_yaml(calibration_targets_path())["apriltags"]
K, D = np.array(info.k).reshape(3, 3), np.array(info.d, float) if len(info.d) else np.zeros(5)
tags = detect_tags(gray, a["dictionary"], a["size_m"], K, D, with_corners=True)
ch = 3 if m.encoding in ("rgb8", "bgr8") else 4
img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)[:, :, :3]
vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if m.encoding.startswith("rgb") else img.copy()
for tid, (T, c) in tags.items():
    cv2.polylines(vis, [c.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.circle(vis, tuple(c[0].astype(int)), 5, (0, 0, 255), -1)          # the tag's TOP-LEFT corner
    cv2.putText(vis, str(tid), tuple(c.mean(axis=0).astype(int) + [-6, 6]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite("/repo/data/overhead_view.png", vis)
want = sorted(int(t["id"]) for t in a["tags"])
print(f"{m.width}x{m.height} {m.encoding}, fx {K[0, 0]:.1f}, distortion {np.round(D, 4).tolist()}")
print(f"tags seen: {sorted(tags)}; missing: {[t for t in want if t not in tags]}; unknown ids: {[t for t in tags if t not in want]}")
for tid, (T, c) in sorted(tags.items()):
    side = np.linalg.norm(c[1] - c[0])
    print(f"  tag {tid}: {side:.0f} px across, {np.linalg.norm(T[:3, 3]):.2f} m away")
# the camera's height above the board + its tilt, from each tag's own pose (needs no tag positions): the board plane
# in the camera frame = each tag's z axis (into the board) and centre
if tags:
    hs, tilts = [], []
    for tid, (T, c) in sorted(tags.items()):
        n = T[:3, 2]                                  # into the board, camera frame
        hs.append(float(n @ T[:3, 3]))                # the camera's distance from the board plane
        tilts.append(float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))))     # optical axis vs straight down
    print(f"camera: {np.median(hs) * 100:.1f} cm above the board (per tag {np.round(np.array(hs) * 100, 1).tolist()}), "
          f"optical axis {np.median(tilts):.1f} deg from straight down (per tag {np.round(tilts, 1).tolist()})")
