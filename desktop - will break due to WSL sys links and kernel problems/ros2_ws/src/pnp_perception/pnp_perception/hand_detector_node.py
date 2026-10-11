"""Is there a hand in the arm's workspace? (the fixed camera; pnp_core/hand_detect.py)

A convenience stop, NOT a safety system: the SO-101 is weak, and this only spares a hand a bump and the pick a
collision it didn't plan for. RobotInterface (pnp_manipulation) listens: a running arm move is cancelled while a hand
is in the workspace and resumed once it has been clear for ~1 s.

  /safety/hand  std_msgs/String JSON, every processed frame (rate_hz):
                {present: bool, n: hands in the workspace, seen: hands in the image, backend, points: [[x, y, z]...]}
  A listener treats a stale message (> 0.6 s) as "no hand": stop this node and nothing pauses.

Params:
  backend       auto (sim -> sim_marker, real -> mediapipe) | sim_marker | mediapipe
  model         MediaPipe's hand_landmarker.task (default models/mediapipe/hand_landmarker.task)
  radius_m      the workspace: within this of the robot base, horizontally (default 0.45: the reach + margin)
  max_height_m  ... and at most this above the table (0.45)
  rate_hz       10
"""

import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from pnp_core.catalog import load_robot, load_scene, load_yaml, repo_root
from pnp_core.geometry import make_transform, quat_to_matrix
from pnp_core.hand_detect import MediaPipeHands, in_workspace, magenta_blobs, pixels_to_base
from pnp_core.perception_math import depth_to_meters


class HandDetector(Node):
    def __init__(self):
        super().__init__("pnp_hand_detector")
        self.declare_parameter("backend", "auto")
        self.declare_parameter("sim", True)
        self.declare_parameter("model", str(repo_root() / "models/mediapipe/hand_landmarker.task"))
        self.declare_parameter("radius_m", 0.45)
        self.declare_parameter("max_height_m", 0.45)
        self.declare_parameter("rate_hz", 10.0)
        backend = self.get_parameter("backend").value
        if backend == "auto":
            backend = "sim_marker" if self.get_parameter("sim").value else "mediapipe"
        self.backend, self.detect = backend, None
        if backend == "sim_marker":
            self.detect = magenta_blobs
        else:
            try:
                self.detect = MediaPipeHands(self.get_parameter("model").value)
            except Exception as e:  # noqa: BLE001  (not installed / no model: report it, publish nothing)
                self.get_logger().error(f"MediaPipe hand detector unavailable ({e}): nothing will pause. "
                                        "pip install mediapipe into /opt/pnp_venv + models/mediapipe/hand_landmarker.task")
        t = load_scene()["table"]
        self.table_z = t["center"][2] + t["size"][2] / 2
        self.base = load_robot()["frames"]["base"]
        topics = load_yaml("config/interfaces.yaml")["topics"]
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.img = self.depth = self.info = None
        self.create_subscription(Image, topics["color_image"]["name"], lambda m: setattr(self, "img", m),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, topics["aligned_depth"]["name"], lambda m: setattr(self, "depth", m),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, topics["color_info"]["name"], lambda m: setattr(self, "info", m),
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(String, "/safety/hand", 10)
        self.present, self.last_img = False, None
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)
        self.get_logger().info(f"hand detector: {self.backend}, workspace r <= {self.get_parameter('radius_m').value} m")

    def _rgb(self, m):
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[m.encoding]
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)
        a = a[:, :, :3]
        return a[:, :, ::-1] if m.encoding.startswith("bgr") else a

    def _tick(self):
        if self.detect is None or self.img is None or self.info is None or self.img is self.last_img:
            return
        m, self.last_img = self.img, self.img
        hands = self.detect(self._rgb(m))
        points, n_in = [], 0
        if hands:
            K = np.array(self.info.k).reshape(3, 3)
            depth, T = None, None
            if self.depth is not None:
                d = self.depth
                depth = depth_to_meters(np.frombuffer(d.data, np.uint16 if d.encoding in ("16UC1", "mono16")
                                                      else np.float32).reshape(d.height, d.width), d.encoding)
            try:
                tf = self.tf.lookup_transform(self.base, m.header.frame_id, Time()).transform
                T = make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                                   [tf.translation.x, tf.translation.y, tf.translation.z])
            except Exception:  # noqa: BLE001
                pass
            for px in hands:
                p = pixels_to_base(px, depth, K, T) if depth is not None and T is not None else None
                inside = in_workspace(p, self.table_z, float(self.get_parameter("radius_m").value),
                                      float(self.get_parameter("max_height_m").value))
                n_in += inside
                points.append(None if p is None else [round(float(v), 3) for v in p])
        present = n_in > 0
        self.pub.publish(String(data=json.dumps({"present": present, "n": n_in, "seen": len(hands),
                                                 "backend": self.backend, "points": points, "t": time.time()})))
        if present != self.present:
            self.present = present
            self.get_logger().info(f"hand in the workspace: {present} {points}")


def main():
    rclpy.init()
    n = HandDetector()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
