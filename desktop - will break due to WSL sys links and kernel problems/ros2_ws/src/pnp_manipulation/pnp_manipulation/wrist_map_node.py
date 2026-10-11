"""The desk flat map from the WRIST camera: the arm looks from a few poses, the depth images are fused into a
top-down height + color grid, and the objects on the desk come out of it (docs/learning/21_wrist_map.md).

    ros2 service call /wrist_map/scan std_srvs/srv/Trigger            (bringup starts this node with a wrist camera)
    ros2 service call /wrist_map/look std_srvs/srv/Trigger            ONE view from where the arm is now, no motion
                                                                      (first real tests: pose the arm by hand)

For a cell with no fixed camera over the desk (the real SO-101 + D405 at first): this is how the robot sees the desk.
Per view: move (joint goal), wait until the arm has settled, take a depth frame captured AFTER that, put its points in
the base frame with TF AT THE IMAGE TIME (the camera moves with the hand: arm kinematics x hand-eye calibration),
drop the strip the jaws cover, fuse (pnp_core/height_map.py). Views: chosen once per run (pnp_core/wrist_scan.py).

Out: /wrist_map/topview (sensor_msgs/Image: the map as a picture, x up), /wrist_map/objects (std_msgs/String JSON,
latched: each object's center, footprint, yaw, height, color, and which catalog object + face it matches),
/wrist_map/markers (RViz boxes).
"""

import json
import threading
import time
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from pnp_core.catalog import load_robot, load_scene, load_yaml, repo_root
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix, rot_z
from pnp_core.height_map import HeightMap, classify, fit_desk_plane, level_to_plane
from pnp_core.perception_math import depth_to_meters
from pnp_core.reorient import rest_poses
from pnp_core.wrist import wrist_camera
from pnp_core.wrist_scan import choose_views
from pnp_manipulation.robot_interface import RobotInterface


class WristMapNode(Node):
    def __init__(self):
        super().__init__("pnp_wrist_map")
        self.declare_parameter("res_m", 0.002)
        self.declare_parameter("settle_s", 0.4)            # images this long after the arm stopped (sim lag, wobble)
        self.declare_parameter("max_views", 8)
        self.declare_parameter("stride", 1)                # every n-th depth pixel (2 mm cells need >= 3 points each)
        self.declare_parameter("jaw_strip", 0.33)          # the bottom fraction of the wrist image the jaws cover
        self.declare_parameter("seed", 3)
        # look AND scan (per view): heights from the desk plane FITTED in the view, not from the configured desk height: an
        # uncalibrated wrist camera sees the desk shifted and tilted (the fit's offset/tilt is logged: a direct read
        # of the calibration error). x/y are not corrected by it.
        self.declare_parameter("look_level_to_desk", True)
        self.cfg, self.scene = load_robot(), load_scene()
        self.wrist = wrist_camera(self.cfg)
        if self.wrist is None:
            raise RuntimeError("this robot has no wrist camera (robot profile wrist_camera)")
        self.base, self.tcp = self.cfg["frames"]["base"], self.cfg["frames"]["tcp"]
        self.optical = self.wrist["frames"]["color_optical"]
        t = self.scene["table"]
        self.table_z = t["center"][2] + t["size"][2] / 2
        # the area to map: where objects + obstacles can be (the scene's obstacle region), on the desk
        reg = self.scene.get("obstacles", {}).get("region") or self.scene["target"]["spawn_region"]
        dx, dy = (t["center"][0] - t["size"][0] / 2, t["center"][0] + t["size"][0] / 2), \
                 (t["center"][1] - t["size"][1] / 2, t["center"][1] + t["size"][1] / 2)
        self.region = (max(reg["x"][0], dx[0]), min(reg["x"][1], dx[1]), max(reg["y"][0], dy[0]), min(reg["y"][1], dy[1]))
        self.catalog = load_yaml("config/objects.yaml")
        # what a blob can be: the scene's objects + every LEGO brick in the catalog (a real desk has whatever bricks
        # are at hand; their sizes tell them apart: 2x2 / 2x3 / 2x4 differ by 8 mm)
        scene_objs = [self.scene["target"]["object"]] + [e["object"] for e in self.scene.get("extra_objects", [])]
        self.names = list(dict.fromkeys(scene_objs + [k for k, v in self.catalog.items() if "brick" in v]))
        self.bridge = CvBridge()
        self.tf = Buffer()
        TransformListener(self.tf, self)
        cb = ReentrantCallbackGroup()
        self.depths, self.colors, self.info = deque(maxlen=10), deque(maxlen=10), None
        tp = self.wrist["topics"]
        self.create_subscription(Image, tp["aligned_depth"], lambda m: self.depths.append(m), qos_profile_sensor_data,
                                 callback_group=cb)
        self.create_subscription(Image, tp["color_image"], lambda m: self.colors.append(m), qos_profile_sensor_data,
                                 callback_group=cb)
        self.create_subscription(CameraInfo, tp["color_info"], lambda m: setattr(self, "info", m),
                                 qos_profile_sensor_data, callback_group=cb)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_img = self.create_publisher(Image, "/wrist_map/topview", latched)
        self.pub_obj = self.create_publisher(String, "/wrist_map/objects", latched)
        self.pub_mk = self.create_publisher(MarkerArray, "/wrist_map/markers", latched)
        # a LIGHT colored cloud of what the wrist camera sees (every `preview_step`-th pixel, `preview_hz`), for RViz on
        # the far side of the link: the raw depth is too heavy for Wi-Fi. 0 Hz: off.
        self.declare_parameter("preview_hz", 2.0)
        self.declare_parameter("preview_step", 4)
        self.pub_cloud = self.create_publisher(PointCloud2, "/wrist_map/cloud", 2)
        # the camera itself, drawn in the BASE frame (axes at the calibrated optical center, viewing pyramid, footprint on
        # the desk, label): RViz on the PC can't build it from TF (the camera frames don't cross the link)
        self.pub_sensors = self.create_publisher(MarkerArray, "/sensors/markers", 2)
        self.cal_label = "?"
        self.create_subscription(String, "/calibration/state", self._on_cal_state, latched)
        if float(self.get_parameter("preview_hz").value) > 0:
            self.create_timer(1.0 / float(self.get_parameter("preview_hz").value), self._preview,
                              callback_group=MutuallyExclusiveCallbackGroup())
        self.R = RobotInterface(self, self.cfg)
        # the robot side owns the desk in MoveIt's world (on the Jetson nobody else adds it; the PC's pick node adds
        # the same box under the same id): top 4 mm below the surface, the base stands on it
        self.declare_parameter("add_desk", True)
        if self.get_parameter("add_desk").value:
            threading.Thread(target=self._add_desk, daemon=True).start()
        self.views, self.busy = None, False
        one = MutuallyExclusiveCallbackGroup()          # scan and look never run at the same time
        self.create_service(Trigger, "/wrist_map/scan", lambda q, r: self._run(self.scan, r), callback_group=one)
        self.create_service(Trigger, "/wrist_map/look", lambda q, r: self._run(self.look, r), callback_group=one)
        self.get_logger().info(f"wrist map: region x {self.region[0]:.2f}..{self.region[1]:.2f}, "
                               f"y {self.region[2]:.2f}..{self.region[3]:.2f} m; call /wrist_map/scan "
                               f"(or /wrist_map/look: one view, no motion)")

    def _add_desk(self):
        if not self.R.wait_ready(timeout=120.0):
            self.get_logger().warn("move_group not ready: desk NOT added to the planning scene")
            return
        t = self.scene["table"]
        c = list(t["center"])
        c[2] = self.table_z - 0.004 - t["size"][2] / 2
        try:
            self.R.add_box("table", c, t["size"])
            self.get_logger().info(f"desk added to MoveIt's world (top {(self.table_z - 0.004) * 1000:.0f} mm)")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"desk NOT added: {e}")

    # --- helpers ---------------------------------------------------------------------------------------------------
    def _T(self, parent, child, stamp=None, timeout=1.0):
        t = self.tf.lookup_transform(parent, child, Time() if stamp is None else Time.from_msg(stamp),
                                     timeout=Duration(seconds=timeout)).transform
        return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                              [t.translation.x, t.translation.y, t.translation.z])

    def _plan_views(self):
        end = time.time() + 15.0
        while (self.R.ik_chain is None or self.info is None) and time.time() < end:
            time.sleep(0.2)
        if self.R.ik_chain is None or self.info is None:
            raise RuntimeError("no kinematic chain / no wrist camera_info yet")
        ch = self.R.ik_chain
        T_tcp_opt = self._T(self.tcp, self.optical, timeout=5.0)
        K = np.array(self.info.k, float).reshape(3, 3)
        d = self.wrist.get("depth", {})
        keep = (0.0, 0.0, 1.0, 1.0 - float(self.get_parameter("jaw_strip").value))
        views, cov = choose_views(lambda q: ch.fk(q) @ T_tcp_opt, ch.lower, ch.upper, self.region, K,
                                  (self.info.width, self.info.height), self.table_z,
                                  np.random.default_rng(int(self.get_parameter("seed").value)),
                                  dist=(max(0.1, 1.5 * float(d.get("min_m", 0.07))), 0.32), keep=keep,
                                  max_range=float(d.get("max_m", 0.5)), fk_tcp=ch.fk,
                                  max_views=int(self.get_parameter("max_views").value))
        self.get_logger().info(f"scan plan: {len(views)} views covering {cov * 100:.0f}% of the region")
        return views, cov

    def _fresh_frame(self, after_ns, timeout=4.0):
        end = time.time() + timeout
        while time.time() < end:
            d = next((m for m in reversed(self.depths) if Time.from_msg(m.header.stamp).nanoseconds > after_ns), None)
            if d is not None:
                st = Time.from_msg(d.header.stamp).nanoseconds
                c = min(self.colors, key=lambda m: abs(Time.from_msg(m.header.stamp).nanoseconds - st), default=None)
                if c is not None and abs(Time.from_msg(c.header.stamp).nanoseconds - st) < 5e7:
                    return d, c
            time.sleep(0.03)
        return None, None

    def _points(self, depth_msg, color_msg):
        dt = np.uint16 if depth_msg.encoding == "16UC1" else np.float32
        depth = depth_to_meters(np.frombuffer(depth_msg.data, dt).reshape(depth_msg.height, depth_msg.width),
                                depth_msg.encoding)
        color = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        s = int(self.get_parameter("stride").value)
        rows = int(depth.shape[0] * (1.0 - float(self.get_parameter("jaw_strip").value)))
        vs, us = np.mgrid[0:rows:s, 0:depth.shape[1]:s]
        z = depth[0:rows:s, ::s]
        dcfg = self.wrist.get("depth", {})
        ok = (z > float(dcfg.get("min_m", 0.07)) * 1.2) & (z < float(dcfg.get("max_m", 0.5)))
        k = self.info.k
        pc = np.column_stack([(us[ok] - k[2]) * z[ok] / k[0], (vs[ok] - k[5]) * z[ok] / k[4], z[ok]])
        T = self._T(self.base, depth_msg.header.frame_id or self.optical, depth_msg.header.stamp)
        cols = color[vs[ok], us[ok]] if color.shape[:2] == depth.shape[:2] else None
        return pc @ T[:3, :3].T + T[:3, 3], cols

    # --- the scan ---------------------------------------------------------------------------------------------------
    def _run(self, fn, resp):
        if self.busy:
            resp.success, resp.message = False, "already scanning"
            return resp
        self.busy = True
        try:
            resp.success, resp.message = True, fn()
        except Exception as e:  # noqa: BLE001
            resp.success, resp.message = False, f"{type(e).__name__}: {e}"
            self.get_logger().error(resp.message)
        finally:
            self.busy = False
        return resp

    def scan(self) -> str:
        t0 = time.time()
        if not self.R.wait_ready(timeout=10.0):
            raise RuntimeError("robot not ready")
        if self.views is None:
            self.views, _ = self._plan_views()
        hm = HeightMap(self.region[:2], self.region[2:], float(self.get_parameter("res_m").value), self.table_z)
        used = 0
        for i, q in enumerate(self.views):
            code = self.R.move_joints(q)
            if code != 1:
                self.get_logger().warn(f"view {i + 1}: move failed ({code}), skipped")
                continue
            after = self.get_clock().now().nanoseconds + int(float(self.get_parameter("settle_s").value) * 1e9)
            d, c = self._fresh_frame(after)
            if d is None:
                self.get_logger().warn(f"view {i + 1}: no fresh wrist frame, skipped")
                continue
            pts, cols = self._points(d, c)
            if bool(self.get_parameter("look_level_to_desk").value):       # per view: each pose has its own error
                fit = fit_desk_plane(pts, self.table_z)
                if fit is not None:
                    pts = level_to_plane(pts, fit[0], fit[1], self.table_z)
            n = hm.add(pts, cols)
            used += 1
            self.get_logger().info(f"view {i + 1}/{len(self.views)}: {len(pts)} points -> {n} cells, "
                                   f"coverage {hm.coverage * 100:.0f}%")
        self.R.move_joints(self.cfg["arm"]["home"])
        return self._finish(hm, f"{used}/{len(self.views)} views", t0, "scan")

    def look(self) -> str:
        """One view from the CURRENT pose; the arm doesn't move (no MoveIt needed: joint states, TF and the camera).
        For a first real test the arm is posed by hand (torque off, or on to hold it), the camera looking down at the
        desk from 15-30 cm. The map only covers what that one view sees."""
        t0 = time.time()
        end = time.time() + 5.0
        while self.info is None and time.time() < end:
            time.sleep(0.1)
        if self.info is None:
            raise RuntimeError("no wrist camera_info: is the D405 streaming?")
        d, c = self._fresh_frame(self.get_clock().now().nanoseconds + int(0.1e9))
        if d is None:
            raise RuntimeError("no fresh wrist depth + color pair (aligned depth on? see realsense_d405.yaml)")
        hm = HeightMap(self.region[:2], self.region[2:], float(self.get_parameter("res_m").value), self.table_z)
        pts, cols = self._points(d, c)
        lvl = ""
        if bool(self.get_parameter("look_level_to_desk").value):
            fit = fit_desk_plane(pts, self.table_z)
            if fit is None:
                lvl = ", desk plane NOT found (heights from the configured desk)"
            else:
                nrm, dd, frac = fit
                ctr = np.array([(self.region[0] + self.region[1]) / 2, (self.region[2] + self.region[3]) / 2])
                z_at = -(dd + nrm[0] * ctr[0] + nrm[1] * ctr[1]) / nrm[2]
                tilt = np.degrees(np.arccos(min(1.0, nrm[2])))
                self.get_logger().info(f"look: desk plane {(z_at - self.table_z) * 1000:+.1f} mm vs configured, tilted "
                                       f"{tilt:.1f} deg ({frac * 100:.0f}% inliers): heights measured from it")
                lvl = f", desk {(z_at - self.table_z) * 1000:+.0f} mm / {tilt:.1f} deg off (levelled)"
                pts = level_to_plane(pts, nrm, dd, self.table_z)
        n = hm.add(pts, cols)
        T = self._T(self.base, self.optical, d.header.stamp)
        self.get_logger().info(f"look: camera at {np.round(T[:3, 3], 3).tolist()} (base), {len(pts)} points -> {n} cells")
        return self._finish(hm, f"1 view (camera at z {T[2, 3]:.2f} m){lvl}", t0, "look")

    def _finish(self, hm, what, t0, tag) -> str:
        objs = hm.objects()
        for o in objs:
            o.match = classify(o, self.catalog, self.names, rest_poses)
        self._publish(hm, objs)
        found = "; ".join(f"{o.match.get('object', '?')} at ({o.center_xy[0]:.3f}, {o.center_xy[1]:.3f}) "
                          f"{o.footprint[0] * 1000:.0f}x{o.footprint[1] * 1000:.0f}x{o.height * 1000:.0f} mm" for o in objs)
        msg = (f"{what}, coverage {hm.coverage * 100:.0f}%, {len(objs)} objects"
               + (f": {found}" if objs else "") + f" in {time.time() - t0:.0f} s")
        self.get_logger().info(f"{tag}: " + msg)
        return msg

    def _on_cal_state(self, msg):
        try:
            self.cal_label = str(json.loads(msg.data).get("d405", "?"))
        except ValueError:
            pass

    def _sensor_markers(self, T, K, size, stamp, ns="d405", label="D405", near=0.07, far=0.30,
                        rgb=(0.1, 0.8, 1.0), strip=0.0) -> list:
        """Markers for one camera (T = base <- optical, K, (w, h)): axes, pyramid near..far, the desk footprint (and the
        image line below which the jaws hide the desk), a label. Pure geometry: reused for the fixed camera later."""
        w, h = size
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        rays = np.array([[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0] for u, v in corners])

        def to_base(p_opt):
            return T[:3, :3] @ p_opt + T[:3, 3]

        def pt(p):
            return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))

        def mk(kind, mid, scale, color, alpha=1.0):
            m = Marker(type=kind, id=mid, ns=ns)
            m.header.frame_id, m.header.stamp = self.base, stamp
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = (float(v) for v in scale)          # ROS wants floats, not 0
            m.color.r, m.color.g, m.color.b, m.color.a = (float(v) for v in (*color, alpha))
            return m

        out = []
        apex = T[:3, 3]
        pyr = mk(Marker.LINE_LIST, 1, (0.0015, 0, 0), rgb, 0.9)
        for i in range(4):
            a, b = rays[i], rays[(i + 1) % 4]
            pyr.points += [pt(apex), pt(to_base(a * far))]
            pyr.points += [pt(to_base(a * far)), pt(to_base(b * far)), pt(to_base(a * near)), pt(to_base(b * near))]
        out.append(pyr)
        # where the corner rays meet the desk: what this view covers (and the jaws' edge, for the wrist camera)
        foot = mk(Marker.LINE_STRIP, 2, (0.002, 0, 0), rgb, 1.0)
        hits = []
        for r in list(rays) + [rays[0]]:
            d = T[:3, :3] @ r
            if d[2] < -1e-6:
                hits.append(apex + d * (self.table_z - apex[2]) / d[2])
        if len(hits) == 5:
            foot.points = [pt(p) for p in hits]
            out.append(foot)
        if strip > 0:
            jaw = mk(Marker.LINE_STRIP, 3, (0.0015, 0, 0), (1.0, 0.5, 0.0), 1.0)
            vj = h * (1.0 - strip)
            for u in (0, w):
                d = T[:3, :3] @ np.array([(u - K[0, 2]) / K[0, 0], (vj - K[1, 2]) / K[1, 1], 1.0])
                if d[2] < -1e-6:
                    jaw.points.append(pt(apex + d * (self.table_z - apex[2]) / d[2]))
            if len(jaw.points) == 2:
                out.append(jaw)
        for k, col in enumerate(((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.2, 0.4, 1.0))):     # optical x y z = R G B
            ax = mk(Marker.ARROW, 10 + k, (0.003, 0.006, 0.008), col)
            ax.points = [pt(apex), pt(apex + T[:3, k] * 0.04)]
            out.append(ax)
        txt = mk(Marker.TEXT_VIEW_FACING, 20, (0, 0, 0.012), (1.0, 1.0, 1.0))
        txt.pose.position = pt(apex + np.array([0, 0, 0.03]))
        txt.text = label
        out.append(txt)
        return out

    def _preview(self):
        if self.info is None or not self.depths or not self.colors:
            return
        d, c = self.depths[-1], self.colors[-1]
        dt = np.uint16 if d.encoding == "16UC1" else np.float32
        depth = depth_to_meters(np.frombuffer(d.data, dt).reshape(d.height, d.width), d.encoding)
        color = self.bridge.imgmsg_to_cv2(c, "bgr8")
        s = int(self.get_parameter("preview_step").value)
        vs, us = np.mgrid[0:depth.shape[0]:s, 0:depth.shape[1]:s]
        z = depth[::s, ::s]
        ok = (z > 0.07) & (z < 0.6)
        k = self.info.k
        xyz = np.column_stack([(us[ok] - k[2]) * z[ok] / k[0], (vs[ok] - k[5]) * z[ok] / k[4], z[ok]])
        # in the BASE frame, transformed here where the TF tree is complete: the camera's frames don't make it across
        # the zenoh link (only one of the two latched /tf_static publishers does)
        try:
            T = self._T(self.base, d.header.frame_id or self.optical, d.header.stamp, timeout=0.2)
        except Exception:  # noqa: BLE001
            return
        xyz = (xyz @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        if color.shape[:2] == depth.shape[:2]:
            bgr = color[vs[ok], us[ok]].astype(np.uint32)
        else:
            bgr = np.full((len(xyz), 3), 200, np.uint32)
        rgb = ((bgr[:, 2] << 16) | (bgr[:, 1] << 8) | bgr[:, 0]).view(np.float32)
        data = np.column_stack([xyz, rgb]).astype(np.float32)
        m = PointCloud2(height=1, width=len(data), is_bigendian=False, point_step=16, row_step=16 * len(data),
                        is_dense=True, data=data.tobytes())
        m.header.stamp, m.header.frame_id = d.header.stamp, self.base
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(("x", "y", "z", "rgb"))]
        self.pub_cloud.publish(m)
        K = np.array(self.info.k, float).reshape(3, 3)
        self.pub_sensors.publish(MarkerArray(markers=self._sensor_markers(
            T, K, (self.info.width, self.info.height), d.header.stamp,
            label=f"D405 wrist ({self.cal_label})", strip=float(self.get_parameter("jaw_strip").value))))

    def _publish(self, hm, objs):
        img = hm.render(scale=2, objects=objs)
        m = self.bridge.cv2_to_imgmsg(img, "bgr8")
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), self.base
        self.pub_img.publish(m)
        data = {"time": time.time(), "region": self.region, "res_m": hm.res, "coverage": round(hm.coverage, 3),
                "objects": [{"xy": [round(v, 4) for v in o.center_xy], "footprint": [round(v, 4) for v in o.footprint],
                             "yaw": round(o.yaw, 4), "height": round(o.height, 4), "color_bgr": list(o.color_bgr),
                             "cells": o.cells, **o.match} for o in objs]}
        self.pub_obj.publish(String(data=json.dumps(data)))
        ma = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, o in enumerate(objs):
            mk = Marker(type=Marker.CUBE, id=i + 1, ns="wrist_map")
            mk.header.frame_id = self.base
            mk.pose.position.x, mk.pose.position.y = o.center_xy
            mk.pose.position.z = self.table_z + o.height / 2
            qx, qy, qz, qw = matrix_to_quat(rot_z(o.yaw))
            mk.pose.orientation.x, mk.pose.orientation.y, mk.pose.orientation.z, mk.pose.orientation.w = qx, qy, qz, qw
            mk.scale.x, mk.scale.y, mk.scale.z = o.footprint[0], o.footprint[1], max(o.height, 0.002)
            b, g, r = (v / 255.0 for v in o.color_bgr)
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 0.8
            ma.markers.append(mk)
            # + the catalog object's own mesh where it was found (upright: yaw only; the boxes stay as the measured
            # outline). The mesh origin is its bounding-box center, z up (tools/make_brick_meshes.py).
            name = o.match.get("object")
            mesh = repo_root() / "models" / "foundationpose" / "meshes" / str(name) / "mesh.obj"
            if name and o.match.get("rest") in ("upright", None):
                mm = Marker(type=Marker.MESH_RESOURCE, id=1000 + i, ns="wrist_map_models")
                mm.header.frame_id = self.base
                mm.mesh_resource, mm.mesh_use_embedded_materials = f"file://{mesh}", True
                h = float(self.catalog[name]["dims_m"][2])
                mm.pose.position.x, mm.pose.position.y, mm.pose.position.z = o.center_xy[0], o.center_xy[1], self.table_z + h / 2
                mm.pose.orientation = mk.pose.orientation
                mm.scale.x = mm.scale.y = mm.scale.z = 1.0
                mm.color.a = 0.0          # 0 + embedded materials: the mesh's own colors
                ma.markers.append(mm)
        self.pub_mk.publish(ma)


def main():
    rclpy.init()
    node = WristMapNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        while rclpy.ok():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()
