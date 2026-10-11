"""One overhead RGB-D frame -> what the detector's size-class path sees (DepthBlobs + CatalogSize): every blob's
footprint, height and identity. Debug aid for perception.detector.size_classes (the 2x3 brick has no YOLO class).

    docker exec pnp_dev bash /repo/data/rospy.sh /repo/data/debug_size_classes.py
"""
import sys
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_camera, load_robot, load_scene, load_yaml  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from pnp_core.perception_math import depth_to_meters  # noqa: E402
from pnp_core.vision import CatalogSize, DepthBlobs, Frame  # noqa: E402

rclpy.init()
n = rclpy.create_node("debug_size_classes")
topics = load_yaml("config/interfaces.yaml")["topics"]
got = {}
n.create_subscription(Image, topics["aligned_depth"]["name"], lambda m: got.setdefault("d", m), qos_profile_sensor_data)
n.create_subscription(CameraInfo, topics["color_info"]["name"], lambda m: got.setdefault("k", m), qos_profile_sensor_data)
tf = Buffer()
TransformListener(tf, n)
end = time.time() + 10
while time.time() < end and not ("d" in got and "k" in got and tf.can_transform(
        load_robot()["frames"]["base"], load_camera()["frames"]["color_optical"], rclpy.time.Time())):
    rclpy.spin_once(n, timeout_sec=0.1)
t = tf.lookup_transform(load_robot()["frames"]["base"], load_camera()["frames"]["color_optical"], rclpy.time.Time())
q, p = t.transform.rotation, t.transform.translation
T = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [p.x, p.y, p.z])
m = got["d"]
depth = depth_to_meters(CvBridge().imgmsg_to_cv2(m, "passthrough"), m.encoding)
sc = load_scene()
tb = sc["table"]
table_z = tb["center"][2] + tb["size"][2] / 2
cx, cy, sx, sy = tb["center"][0], tb["center"][1], tb["size"][0], tb["size"][1]
rr = float(sc.get("ui", {}).get("reach_m", [0.0, 0.3])[1]) + 0.03
region = (0.0, rr, -rr, rr)   # as detector_node
print("depth", depth.shape, "valid", float((depth > 0).mean()), "table_z", table_z, "region", np.round(region, 3))
frame = Frame(None, depth, np.asarray(got["k"].k, float).reshape(3, 3), T)
sizer = CatalogSize(load_yaml("config/objects.yaml"))
for stride in (1,):
    print(f"stride {stride}")
    t0 = time.time()
    blobs = DepthBlobs(region, table_z, stride=stride).propose(frame)
    print(f"  propose: {(time.time() - t0) * 1000:.0f} ms")
    for b in blobs:
        ident = sizer.identify(frame, b)
        print(f"  blob at ({b.center_xy[0]:.3f}, {b.center_xy[1]:.3f}) footprint {np.round(np.array(b.footprint) * 1000, 1)} "
              f"mm height {b.height * 1000:.1f} mm -> {ident.name if ident else None} {ident.extra if ident else ''}")
