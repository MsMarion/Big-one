"""Close look from wrist DEPTH (no learned model): the /perception/refine_pose service for objects FoundationPose has
no mesh for, e.g. the SO-101's LEGO bricks (scene key perception.wrist_refine: depth; bringup starts this node).

pick_place_node calls the service at the pre-grasp pose (its LOOK step, param wrist_refine); this node waits for a
wrist depth frame taken after the request, turns it into base-frame points (TF at the image time: the camera moves
with the hand) and runs pnp_core.wrist_refine.refine_on_desk. The answer has the same form as FoundationPose's
(pose_estimator_node.py), so the pick logic is the same: it keeps the first estimate if the correction is too big.
docs/learning/17_so101.md "Wrist camera".
"""

import threading
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from pnp_core.focus import parse_focus  # noqa: E402
from pnp_core.catalog import load_object, load_robot, load_scene
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix
from pnp_core.perception_math import depth_to_meters
from pnp_core.wrist import wrist_camera
from pnp_core.wrist_refine import refine_on_desk
from pnp_interfaces.srv import RefinePose


class WristRefineNode(Node):
    def __init__(self):
        super().__init__("wrist_refine_node")
        self.declare_parameter("settle_s", 0.15)        # the image must be this much newer than the request (sim lag)
        self.declare_parameter("stride", 2)             # use every n-th depth pixel (640x360 -> ~57k points)
        robot, scene = load_robot(), load_scene()
        self.base = robot["frames"]["base"]
        self.wrist = wrist_camera(robot)
        self.obj = load_object(scene["target"]["object"])
        t = scene["table"]
        self.desk_z = float(t["center"][2] + t["size"][2] / 2)
        self.depth_range = (float(self.wrist["depth"]["min_m"]) * 0.8, float(self.wrist["depth"]["max_m"]))
        self.tf = Buffer()
        TransformListener(self.tf, self)
        cb = ReentrantCallbackGroup()
        self.depths, self.info, self.lock = deque(maxlen=5), None, threading.Lock()
        topics = self.wrist["topics"]
        self.create_subscription(Image, topics["aligned_depth"], lambda m: self.depths.append(m), qos_profile_sensor_data,
                                 callback_group=cb)
        self.create_subscription(CameraInfo, topics["color_info"], lambda m: setattr(self, "info", m),
                                 qos_profile_sensor_data, callback_group=cb)
        self.create_service(RefinePose, "/perception/refine_pose", self._refine,
                            callback_group=MutuallyExclusiveCallbackGroup())
        # several objects on the table: refine the one being picked (pick_place_node publishes it)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/perception/focus", self._on_focus, latched)
        self.get_logger().info(f"close look from wrist depth ({self.wrist['name']}) for {scene['target']['object']}")

    def _on_focus(self, msg):
        try:
            self.obj = load_object(parse_focus(msg.data)[0])
        except KeyError:
            self.get_logger().warn(f"focus '{msg.data}': not in config/objects.yaml")

    def _T_base(self, frame, stamp):
        try:
            tf = self.tf.lookup_transform(self.base, frame, Time.from_msg(stamp), timeout=Duration(seconds=0.3)).transform
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"no TF {self.base} <- {frame}: {e}")
            return None
        return make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                              [tf.translation.x, tf.translation.y, tf.translation.z])

    def _refine(self, req, resp):
        t_req = self.get_clock().now().nanoseconds + int(self.get_parameter("settle_s").value * 1e9)
        q, t = req.initial.pose.orientation, req.initial.pose.position
        if not req.initial.header.frame_id:
            resp.success, resp.message = False, "no initial pose given (this node tracks nothing itself)"
            return resp
        T_init = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])
        end, frame = time.time() + (req.timeout_s or 2.0), None
        while time.time() < end and frame is None:
            frame = next((d for d in reversed(self.depths) if Time.from_msg(d.header.stamp).nanoseconds > t_req), None)
            if frame is None or self.info is None:
                frame = None
                time.sleep(0.03)
        if frame is None:
            resp.success, resp.message = False, f"no fresh wrist depth frame ({len(self.depths)} buffered)"
            return resp
        T_bc = self._T_base(frame.header.frame_id, frame.header.stamp)
        if T_bc is None:
            resp.success, resp.message = False, f"no TF for {frame.header.frame_id} at the image time"
            return resp
        dt = np.uint16 if frame.encoding == "16UC1" else np.float32
        depth = depth_to_meters(np.frombuffer(frame.data, dt).reshape(frame.height, frame.width), frame.encoding)
        s = int(self.get_parameter("stride").value)
        k = self.info.k
        vs, us = np.mgrid[0:frame.height:s, 0:frame.width:s]
        d = depth[::s, ::s]
        ok = (d > self.depth_range[0]) & (d < self.depth_range[1])
        pc = np.column_stack([(us[ok] - k[2]) * d[ok] / k[0], (vs[ok] - k[5]) * d[ok] / k[4], d[ok]])
        t0 = time.time()
        r = refine_on_desk(pc @ T_bc[:3, :3].T + T_bc[:3, 3], T_init, self.obj, desk_z=self.desk_z)
        ms = (time.time() - t0) * 1000
        if r.T_base_obj is None:
            resp.success, resp.message = False, f"{r.message} {r.info}"
            self.get_logger().info(f"close look: {resp.message}")
            return resp
        T = r.T_base_obj
        dR = T_init[:3, :3].T @ T[:3, :3]
        resp.correction_m = float(np.linalg.norm(T[:3, 3] - T_init[:3, 3]))
        resp.correction_deg = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        resp.pose.header.frame_id, resp.pose.header.stamp = self.base, frame.header.stamp
        resp.pose.pose.position.x, resp.pose.pose.position.y, resp.pose.pose.position.z = map(float, T[:3, 3])
        o = resp.pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, matrix_to_quat(T[:3, :3]))
        resp.success = True
        resp.message = (f"wrist depth in {ms:.0f} ms: moved {resp.correction_m * 1000:.1f} mm, "
                        f"{resp.correction_deg:.1f} deg {r.info}")
        self.get_logger().info(f"close look: {resp.message}")
        return resp


def main():
    rclpy.init()
    n = WristRefineNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()
