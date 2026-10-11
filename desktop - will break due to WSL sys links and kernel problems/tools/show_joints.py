"""Print current joint positions/velocities (+ gripper width) from /joint_states.

    python3 /repo/tools/show_joints.py [--topic /isaac_joint_states]
"""
import argparse
import time

import rclpy
from sensor_msgs.msg import JointState

ap = argparse.ArgumentParser()
ap.add_argument("--topic", default="/joint_states")
a = ap.parse_args()
rclpy.init()
n = rclpy.create_node("show_joints")
msg = []
n.create_subscription(JointState, a.topic, lambda m: msg.append(m), 10)
t0 = time.time()
while not msg and time.time() - t0 < 5:
    rclpy.spin_once(n, timeout_sec=0.1)
if not msg:
    print(f"no message on {a.topic}")
else:
    m = msg[-1]
    vel = list(m.velocity) or [float("nan")] * len(m.name)
    for name, p, v in zip(m.name, m.position, vel):
        print(f"{name:20s} pos {p:+.4f}  vel {v:+.4f}")
    d = dict(zip(m.name, m.position))
    if "fr3_finger_joint1" in d:
        print(f"gripper width: {2 * d['fr3_finger_joint1'] * 1000:.1f} mm")
