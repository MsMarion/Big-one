"""Where SHOULD the AprilTags appear (their configured positions, through the sim's TRUE camera pose and the published
camera_info) vs where they ARE detected: the pixel offsets' pattern says what's off.
  mean offset (dx, dy)      -> the principal point (cx, cy) the sim renders with != camera_info
  offsets growing outward   -> the focal length (fx, fy) != camera_info, or the tags' size
  one side only             -> a tag's position / height
    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/tag_projection_check.py
"""
import sys
import time

import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo")
sys.path.insert(0, "/repo/tools")
from pnp_core.calibration import detect_tags, image_to_gray, tag_corners  # noqa: E402
from pnp_core.catalog import calibration_targets_path, load_scene, load_yaml  # noqa: E402
from pnp_core.geometry import make_transform  # noqa: E402
from sim_client import SimClient  # noqa: E402

rclpy.init()
n = rclpy.create_node("tag_projection_check")
got = {}
n.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: got.__setitem__("img", m), qos_profile_sensor_data)
n.create_subscription(CameraInfo, "/camera/camera/color/camera_info", lambda m: got.__setitem__("info", m),
                      qos_profile_sensor_data)
end = time.time() + 5
while time.time() < end and not ("img" in got and "info" in got):
    rclpy.spin_once(n, timeout_sec=0.1)
sim = SimClient(n)
T_true = np.array(sim.call("camera_truth")["cameras"]["camera"])          # base <- optical, the sim's real camera
K = np.array(got["info"].k).reshape(3, 3)
print(f"camera_info: fx {K[0, 0]:.2f} fy {K[1, 1]:.2f} cx {K[0, 2]:.2f} cy {K[1, 2]:.2f}  "
      f"{got['info'].width}x{got['info'].height}")
scene = load_scene()
z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
a = load_yaml(calibration_targets_path())["apriltags"]
seen = detect_tags(image_to_gray(got["img"]), a["dictionary"], a["size_m"], K, np.zeros(5), with_corners=True)
rows = []
for tag in a["tags"]:
    if tag["id"] not in seen:
        continue
    c, s = np.cos(np.radians(tag["sim_yaw_deg"])), np.sin(np.radians(tag["sim_yaw_deg"]))
    for dz in (0.0, 0.0008):
        T = make_transform(np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]]), [tag["sim_xy"][0], tag["sim_xy"][1], z + dz])
        P = tag_corners(a["size_m"]) @ T[:3, :3].T + T[:3, 3]
        p = (P - T_true[:3, 3]) @ T_true[:3, :3]
        uv = (p[:, :2] / p[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        if dz == 0.0008:
            d = seen[tag["id"]][1] - uv
            rows.append((tag["id"], uv.mean(0), d.mean(0), d))
print(f"{'id':>3} {'u':>6} {'v':>6} {'mean dx':>8} {'mean dy':>8}   per-corner (dx, dy)")
for tid, c, m, d in rows:
    print(f"{tid:>3} {c[0]:6.0f} {c[1]:6.0f} {m[0]:8.2f} {m[1]:8.2f}   " + " ".join(f"({x:+.1f},{y:+.1f})" for x, y in d))
allm = np.array([r[2] for r in rows])
print(f"all tags: mean offset ({allm[:, 0].mean():+.2f}, {allm[:, 1].mean():+.2f}) px")
# what that offset does to a one-image pose (what calibrate.py locate solves): PnP over every corner, vs the truth
import cv2  # noqa: E402
from pnp_core.calibration import pose_error  # noqa: E402
P3, P2 = [], []
for tag in a["tags"]:
    if tag["id"] in seen:
        c, s = np.cos(np.radians(tag["sim_yaw_deg"])), np.sin(np.radians(tag["sim_yaw_deg"]))
        T = make_transform(np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]]), [tag["sim_xy"][0], tag["sim_xy"][1], z + 0.0008])
        P3.append(tag_corners(a["size_m"]) @ T[:3, :3].T + T[:3, 3])
        P2.append(seen[tag["id"]][1])
P3, P2 = np.vstack(P3), np.vstack(P2).astype(np.float64)
for label, shift in (("camera_info as published", (0.0, 0.0)), ("+ the measured mean offset", tuple(allm.mean(0))),
                     ("+ a half pixel (-0.5, -0.5)", (-0.5, -0.5))):
    K2 = K.copy()
    K2[0, 2] += shift[0]
    K2[1, 2] += shift[1]
    ok, rv, tv = cv2.solvePnP(P3, P2, K2, np.zeros(5), flags=cv2.SOLVEPNP_SQPNP)
    T_cb = make_transform(cv2.Rodrigues(rv)[0], tv.ravel())
    e = pose_error(np.linalg.inv(T_cb), T_true)
    print(f"PnP with {label:32s}: {e[0]:.2f} mm / {e[1]:.3f} deg from the TRUE camera")
