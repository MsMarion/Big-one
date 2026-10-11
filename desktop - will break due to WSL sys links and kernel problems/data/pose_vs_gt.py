"""FoundationPose's brick pose vs the sim's ground truth (TF 'target'), a few samples."""
import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

import sys
sys.path.insert(0, "/repo")
from pnp_core.catalog import load_object, load_scene  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from pnp_core.object_grasp import rotation_error_deg  # noqa: E402

rclpy.init()
n = Node("pose_vs_gt")
tf = Buffer()
TransformListener(tf, n)
got = {}
n.create_subscription(PoseStamped, "/perception/object_pose", lambda m: got.__setitem__("pose", m), 10)
n.create_subscription(String, "/perception/pose_status", lambda m: got.__setitem__("status", m.data), 10)
obj = load_object(load_scene()["target"]["object"])
end = time.time() + 20
k = 0
while time.time() < end and k < 5:
    rclpy.spin_once(n, timeout_sec=0.2)
    if "pose" in got:
        m = got.pop("pose")
        try:
            t = tf.lookup_transform(m.header.frame_id, "target", Time()).transform
        except Exception as e:  # noqa: BLE001
            print("no GT tf:", e)
            continue
        Tg = make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                            [t.translation.x, t.translation.y, t.translation.z])
        p, q = m.pose.position, m.pose.orientation
        Te = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [p.x, p.y, p.z])
        print(f"pose err {np.linalg.norm(Te[:3, 3] - Tg[:3, 3]) * 1000:.1f} mm, "
              f"{rotation_error_deg(Te[:3, :3], Tg[:3, :3], obj):.1f} deg (symmetry-aware)")
        k += 1
        time.sleep(0.5)
print("status:", json.loads(got["status"]).get("state") if "status" in got else "none", got.get("status", "")[:300])
