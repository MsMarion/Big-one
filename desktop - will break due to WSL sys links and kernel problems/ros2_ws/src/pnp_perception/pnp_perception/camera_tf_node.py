"""Where the cameras are, for everyone: static TF from the calibration files, reloadable live. docs/learning/16_calibration.md

    started by bringup.launch.py;  after a calibration:  ros2 service call /calibration/reload std_srvs/srv/Trigger

Publishes
  fixed camera  fr3_link0 -> camera_link -> camera_color_optical_frame
  wrist camera  fr3_hand  -> wrist_camera_link -> wrist_camera_color_optical_frame   (moves with the hand)
from, per camera, the first that exists:
  config/calibration/<set>_<camera>.yaml   T_parent_optical (hand-eye result of tools/calibrate.py)
  the nominal mount                       config/camera_d435i.yaml mount / config/camera_d405.yaml mount
  (real fixed camera: <cell calibration dir>/d435i_real.yaml may also hold a measured mount or T_base_optical)
<set> = the ACTIVE calibration set (param calibration_set, switchable live): real on hardware; in the sim one set per
reality profile, sim_<profile> (each profile puts the simulated cameras somewhere else: a different "physical setup";
tools/set_reality.py switches it; sim_ideal has no file = the exact nominal mounts).
/calibration/state (std_msgs/String JSON, latched) says which set and which source per camera; the calibration
tools and the drift monitor follow it.
"""

import json
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from pnp_core.catalog import calibration_dir, load_fixed_camera, load_robot, load_scene, load_yaml, repo_root
from pnp_core.geometry import R_LINK_OPTICAL, look_at_optical, make_transform, matrix_to_quat, camera_mount_T, quat_to_matrix
from pnp_core.wrist import T_parent_optical, wrist_camera


def calib_path(cal_set: str, camera: str) -> Path:
    return calibration_dir() / f"{cal_set}_{camera}.yaml"


def load_calibration(cal_set: str, camera: str):
    """-> (T_parent_optical | None, info dict) from a calibration file."""
    p = calib_path(cal_set, camera)
    if not p.exists():
        return None, {}
    c = yaml.safe_load(p.read_text())
    t = c["T_parent_optical"]
    return make_transform(quat_to_matrix(t["quat_xyzw"]), t["xyz"]), c


class CameraTF(Node):
    def __init__(self):
        super().__init__("pnp_camera_tf")
        self.declare_parameter("calibration_set", "sim")
        self.declare_parameter("fixed_camera", True)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        # NOT tf2_ros.StaticTransformBroadcaster: Jazzy's (rclpy) ignores a new transform for a child frame it has
        # sent before, so a reload would republish the OLD pose. We publish the complete set ourselves (same QoS).
        self.pub_tf = self.create_publisher(TFMessage, "/tf_static", latched)
        self.pub_state = self.create_publisher(String, "/calibration/state", latched)
        # the fixed camera's stand for RViz (the scene's camera_rig: a CAD export's meshes, else a pole / boom like the
        # sim draws): /scene/camera_rig_markers, base frame, latched
        self.pub_rig = self.create_publisher(MarkerArray, "/scene/camera_rig_markers", latched)
        self.create_service(Trigger, "/calibration/reload", self._reload)
        self.add_on_set_parameters_callback(self._on_params)
        self._publish()

    def _on_params(self, params):
        if any(p.name in ("calibration_set", "fixed_camera") for p in params):
            t = self.create_timer(0.1, lambda: (self._publish(), t.cancel()))     # after the new value is in place
        return SetParametersResult(successful=True)

    def _cameras(self):
        """-> [(name, parent, link, optical, T_parent_optical, source info)]"""
        cal_set = self.get_parameter("calibration_set").value
        robot = load_robot()
        out = []
        if self.get_parameter("fixed_camera").value:
            cam = load_fixed_camera()
            T, info = load_calibration(cal_set, cam["name"])
            src = f"calibration ({calib_path(cal_set, cam['name']).name}, {info.get('date', '?')})"
            measured = calibration_dir() / "d435i_real.yaml"   # a hand-measured mount of THIS cell's real camera
            if T is None and cal_set == "real" and measured.exists():
                real = load_yaml(str(measured))
                if "T_base_optical" in real:
                    t = real["T_base_optical"]
                    T, src = make_transform(quat_to_matrix(t["quat_xyzw"]), t["xyz"]), "d435i_real.yaml T_base_optical"
                else:
                    m = real["mount"]
                    T = make_transform(look_at_optical(m["position"], m["look_at"]), m["position"])
                    src = "d435i_real.yaml measured mount"
            if T is None:
                m = cam["mount"]
                T, src = camera_mount_T(m), "nominal mount"
            out.append((cam["name"], robot["frames"]["base"], cam["frames"]["link"], cam["frames"]["color_optical"], T, src))
        w = wrist_camera(robot)
        if w:
            T, info = load_calibration(cal_set, w["name"])
            src = f"calibration ({calib_path(cal_set, w['name']).name}, {info.get('date', '?')})" if T is not None \
                else "nominal mount"
            if T is None:
                T = T_parent_optical(w)
            out.append((w["name"], w["mount"]["parent"], w["frames"]["link"], w["frames"]["color_optical"], T, src))
        return out

    def _tf(self, parent, child, T):
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, T[:3, 3])
        q = matrix_to_quat(T[:3, :3])
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = map(float, q)
        return m

    def _publish(self):
        msgs, state = [], {"set": self.get_parameter("calibration_set").value}
        for name, parent, link, optical, T_po, src in self._cameras():
            T_pl = T_po @ make_transform(R_LINK_OPTICAL, [0, 0, 0]).T     # link = optical x (link -> optical)^-1
            msgs += [self._tf(parent, link, T_pl), self._tf(link, optical, make_transform(R_LINK_OPTICAL, [0, 0, 0]))]
            state[name] = {"parent": parent, "source": src, "xyz": np.round(T_po[:3, 3], 5).tolist()}
            self.get_logger().info(f"{name}: {parent} -> {optical} from {src}")
        self.pub_tf.publish(TFMessage(transforms=msgs))
        self.pub_state.publish(String(data=json.dumps(state)))
        try:
            self.pub_rig.publish(self._rig_markers())
        except Exception as e:  # noqa: BLE001  (a missing mesh must not stop the camera TF)
            self.get_logger().warn(f"camera stand markers: {e}")
        return state

    def _marker(self, i, typ, T, scale, rgba, mesh=None):
        m = Marker()
        m.header.frame_id = load_robot()["frames"]["base"]
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = "camera_stand", i, typ, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, T[:3, 3])
        q = matrix_to_quat(T[:3, :3])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, q)
        m.scale.x, m.scale.y, m.scale.z = map(float, scale)
        m.color.r, m.color.g, m.color.b, m.color.a = map(float, rgba)
        if mesh:
            m.mesh_resource = f"file://{mesh}"
        return m

    def _rig_markers(self):
        """The stand as RViz markers, where the sim draws it (sim/scene.py add_urdf_rig / add_camera_rig)."""
        scene = load_scene()
        t = scene["table"]
        top = t["center"][2] + t["size"][2] / 2
        rig = scene.get("camera_rig") or {}
        out = MarkerArray()
        if rig.get("urdf"):
            from pnp_core.urdf_visuals import axis_R, link_T, load_urdf_visuals
            base, joints, visuals = load_urdf_visuals(repo_root() / rig["urdf"])
            q = {name: float(rig.get("tilt_rad", 0.0)) for (_p, _T, _a, typ, name) in joints.values()
                 if typ in ("revolute", "continuous")}
            T_mount = make_transform(axis_R([0, 0, 1], np.radians(float(rig.get("yaw_deg", 0.0)))),
                                     [float(rig["xy"][0]), float(rig["xy"][1]), top])
            i = 0
            for link, items in visuals.items():
                for stl, T_vis, rgba in items:
                    out.markers.append(self._marker(i, Marker.MESH_RESOURCE, T_mount @ link_T(link, joints, q) @ T_vis,
                                                    (1, 1, 1), (rgba or [0.6, 0.6, 0.6, 1.0])[:3] + [1.0], str(stl)))
                    i += 1
            return out
        # no CAD stand: a D435i-sized body at the believed camera pose + a pole (or a boom beside the table)
        cam = load_fixed_camera()
        T_opt = camera_mount_T(cam["mount"])
        T_link = T_opt @ make_transform(R_LINK_OPTICAL, [0, 0, 0]).T        # camera_link: x = viewing direction
        body = T_link @ make_transform(np.eye(3), [-0.0135, 0.0, 0.0])
        out.markers.append(self._marker(0, Marker.CUBE, body, (0.025, 0.090, 0.025), (0.55, 0.57, 0.6, 1.0)))
        pos = T_opt[:3, 3]
        floor = top - float(t.get("height_above_floor", 0.75))
        boom = (cam.get("mount") or {}).get("boom")
        dark = (0.08, 0.08, 0.1, 1.0)
        if boom is None:
            h = pos[2] - 0.03 - floor
            out.markers.append(self._marker(1, Marker.CYLINDER, make_transform(np.eye(3), [pos[0], pos[1], floor + h / 2]),
                                            (0.024, 0.024, h), dark))
        else:
            b = np.asarray(boom, float)
            h = pos[2] + 0.03 - floor
            out.markers.append(self._marker(1, Marker.CYLINDER, make_transform(np.eye(3), [b[0], b[1], floor + h / 2]),
                                            (0.024, 0.024, h), dark))
            d = pos[:2] - b
            yaw = float(np.arctan2(d[1], d[0]))
            c, s = np.cos(yaw), np.sin(yaw)
            mid = [(b[0] + pos[0]) / 2, (b[1] + pos[1]) / 2, pos[2] + 0.03]
            out.markers.append(self._marker(2, Marker.CUBE, make_transform(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), mid),
                                            (float(np.linalg.norm(d)), 0.02, 0.02), dark))
        return out

    def _reload(self, _req, resp):
        state = self._publish()
        resp.success, resp.message = True, json.dumps({k: v["source"] if isinstance(v, dict) else v for k, v in state.items()})
        return resp


def main():
    rclpy.init()
    n = CameraTF()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
