"""A see-through "ghost" of the robot at a given joint pose, next to the live robot in RViz: where to put the arm by
hand, or what a planned pose looks like, without moving anything. (tools/rviz/robot_only.rviz shows it.)

    python3 /repo/tools/show_pose.py 0 0 0 80 0            # degrees: pan lift elbow wrist_flex wrist_roll [gripper]
    python3 /repo/tools/show_pose.py --rad 0 0 0 1.4 0

It reads /robot_description (the live robot's, e.g. from the Jetson), runs a second robot_state_publisher whose TF
frames are prefixed "target/", publishes the pose on /target/joint_states, and pins target/base_link to base_link.
"""
import argparse
import os
import subprocess
import tempfile
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("q", type=float, nargs="+", help="joint values, in the order " + " ".join(ARM))
    ap.add_argument("--rad", action="store_true", help="values are radians (default degrees)")
    a = ap.parse_args()
    q = np.array(a.q + [0.0] * (len(ARM) - len(a.q)))[:len(ARM)]
    q = q if a.rad else np.radians(q)

    rclpy.init()
    n = rclpy.create_node("show_pose")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    got = {}
    n.create_subscription(String, "/robot_description", lambda m: got.setdefault("urdf", m.data), latched)
    end = time.time() + 15
    while "urdf" not in got and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.2)
    if "urdf" not in got:
        raise SystemExit("no /robot_description (is the robot side running and linked?)")
    params = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump({"target_state_publisher": {"ros__parameters": {"robot_description": got["urdf"],
                                                                   "frame_prefix": "target/"}}}, params)
    params.close()
    rsp = subprocess.Popen(["ros2", "run", "robot_state_publisher", "robot_state_publisher", "--ros-args",
                            "-r", "__node:=target_state_publisher", "--params-file", params.name,
                            "-r", "joint_states:=/target/joint_states",
                            "-r", "robot_description:=/target/robot_description"])
    st = StaticTransformBroadcaster(n)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = "base_link", "target/base_link"
    tf.transform.rotation.w = 1.0
    tf.header.stamp = n.get_clock().now().to_msg()
    st.sendTransform(tf)
    pub = n.create_publisher(JointState, "/target/joint_states", 10)
    print("ghost at (deg): " + ", ".join(f"{j} {np.degrees(v):.0f}" for j, v in zip(ARM, q)) + "   Ctrl+C to stop")
    try:
        while rclpy.ok():
            pub.publish(JointState(name=ARM, position=q.tolist(), header=tf.header.__class__(
                stamp=n.get_clock().now().to_msg())))
            rclpy.spin_once(n, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    rsp.terminate()
    os.unlink(params.name)
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
