"""Does a held object turn IN THE HAND (pivot about the fingers) while the hand turns it over? Samples the sim's
ground-truth TF (base -> object frame, base -> TCP) during a workcell task and prints the object's rotation
relative to the hand, from the grasp on, against the task state.

    bash /repo/data/rospy.sh /repo/data/grip_slip_probe.py cracker_box 90       (object frame, seconds)
"""
import json
import sys
import threading
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

sys.path.insert(0, "/repo")
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402

frame, secs = (sys.argv[1] if len(sys.argv) > 1 else "cracker_box"), float(sys.argv[2] if len(sys.argv) > 2 else 90)
rclpy.init()
n = rclpy.create_node("grip_slip_probe")
tf = Buffer()
TransformListener(tf, n)
ex = MultiThreadedExecutor(num_threads=2)
ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()


def T(child):
    t = tf.lookup_transform("fr3_link0", child, Time()).transform
    return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                          [t.translation.x, t.translation.y, t.translation.z])


ref, last = None, None
end = time.time() + secs
while time.time() < end:
    try:
        state = json.load(urllib.request.urlopen("http://localhost:8088/api/world", timeout=2))["task"]["state"]
        Tr = np.linalg.inv(T("fr3_hand_tcp")) @ T(frame)               # the object seen from the hand
    except Exception:  # noqa: BLE001
        time.sleep(0.2)
        continue
    if state in ("LIFT", "MOVE_PREPLACE", "DESCEND", "RELEASE") and ref is None:
        ref = Tr
    if ref is not None and state != last or (ref is not None and state in ("MOVE_PREPLACE", "DESCEND")):
        dR = ref[:3, :3].T @ Tr[:3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        dp = np.linalg.norm(Tr[:3, 3] - ref[:3, 3]) * 1000
        print(f"{state:14s} object turned {ang:5.1f} deg, moved {dp:5.1f} mm in the hand since the lift", flush=True)
        last = state
    if state in ("RETREAT", "HOME") and ref is not None:
        break
    time.sleep(0.25)
ex.shutdown()
rclpy.try_shutdown()
