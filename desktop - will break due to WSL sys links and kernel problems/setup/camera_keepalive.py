"""Keep the bridge's camera routes alive across ROS stack restarts (started by setup/stack.ps1, NOT stopped by
setup/stop_ros.sh).

Why: several times (2026-10-11) Isaac Sim kept simulating but stopped publishing camera images right after the
ROS stack was restarted. A stack restart removes every subscriber of the camera topics in the container, so the
Windows zenoh bridge tears down its DDS readers of Isaac's camera publishers and creates them again a few seconds
later. Suspected: that reader churn on Isaac's side. This node stays subscribed, so the routes never go away.
(Experiment + result: docs/learning/14_wrist_camera.md.)
"""

from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image


def main():
    rclpy.init()
    n = Node("pnp_camera_keepalive")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    t = yaml.safe_load(open("/repo/config/interfaces.yaml"))["topics"]
    topics = [(Image, t["color_image"]["name"]), (Image, t["aligned_depth"]["name"]), (CameraInfo, t["color_info"]["name"])]
    # every robot's wrist camera: this node outlives scene (robot) switches
    for wpath in sorted({yaml.safe_load(p.read_text()).get("wrist_camera") for p in Path("/repo/config").glob("robot_*.yaml")} - {None}):
        w = yaml.safe_load(open(f"/repo/{wpath}"))["topics"]
        topics += [(Image, w["color_image"]), (Image, w["aligned_depth"]), (CameraInfo, w["color_info"])]
    topics = list(dict.fromkeys(topics))
    raw = yaml.safe_load(open("/repo/config/interfaces.yaml")).get("sim_raw_prefix", "")
    topics = [(typ, raw + name) for typ, name in topics]       # sim: Isaac publishes under /sim_raw (sensor model)
    for typ, name in topics:
        n.create_subscription(typ, name, lambda _m: None, qos)
    n.get_logger().info(f"keeping {len(topics)} camera routes alive")
    rclpy.spin(n)


if __name__ == "__main__":
    main()
