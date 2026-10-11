"""Obstacle cloud for MoveIt's Octomap (Phase 5, Layer B): what the depth camera sees, minus robot and targets.

    ros2 run pnp_perception obstacle_cloud_node        (started by bringup.launch.py obstacle_source:=perceived)

  aligned depth + camera_info + TF (camera mount, robot links) + /perception/detections (the cans)
    --pnp_core.cloud.obstacle_cloud-->  /perception/obstacle_cloud  (sensor_msgs/PointCloud2, CAMERA frame, ~5 Hz)
    --> move_group's PointCloudOctomapUpdater (config/sensors_3d.yaml) -> Octomap in the planning scene

Published in the camera frame on purpose: the Octomap updater casts rays from the cloud's frame origin to every
point to CLEAR free space (e.g. where a box was before it moved). In the base frame it would cast them from the
robot base. MoveIt additionally removes the robot itself and every collision object (the table) from the cloud.
"""

import json
import time
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import rclpy
import rclpy.parameter
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import Point
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from pnp_core.focus import nearest, parse_focus  # noqa: E402
from pnp_core.catalog import load_camera, load_object, load_robot, load_yaml
from pnp_core.cloud import (in_cylinders, box_vertices, depth_to_cloud, drop_depth_edges, far_rays, fit_capsule, load_stl_vertices, near_capsule_list,
                            obstacle_cloud, push_beyond)
from pnp_core.geometry import make_transform, quat_to_matrix
from pnp_core.map_monitor import appeared, clusters, key_centers, newly_seen, seen_through, summarize, voxel_keys
from pnp_core.perception_math import depth_to_meters


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def link_capsules(urdf: str) -> dict:
    """{link: (a, b, r)} in each link's frame: one capsule around all of its <collision> geometry.
    The robot must be removed from the cloud BEFORE columnizing (a column under the arm would no longer touch the
    arm, so MoveIt's own self-filter, which runs later, couldn't remove it). Capsules from the real collision
    geometry are tight; the first version used capsules between joint origins, but the FR3 forearm sits ~8 cm off
    that line, and its points became phantom pillars that the arm then "hit" (fr3_link5 vs <octomap>)."""
    out = {}
    for link in ET.fromstring(urdf).iter("link"):
        verts = []
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
            R = _rpy(*[float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])])
            g = col.find("geometry")[0]
            if g.tag == "mesh":
                fn = g.get("filename")
                if fn.startswith("package://"):
                    pkg, rel = fn[len("package://"):].split("/", 1)
                    fn = f"{get_package_share_directory(pkg)}/{rel}"
                elif fn.startswith("file://"):
                    fn = fn[len("file://"):]
                scale = np.array([float(v) for v in g.get("scale", "1 1 1").split()])
                v = load_stl_vertices(fn) * scale
            elif g.tag == "box":
                v = box_vertices([float(x) for x in g.get("size").split()])
            elif g.tag == "cylinder":
                r, h = float(g.get("radius")), float(g.get("length"))
                v = np.array([[r * np.cos(a), r * np.sin(a), z] for a in np.linspace(0, 2 * np.pi, 16)
                              for z in (-h / 2, h / 2)])
            elif g.tag == "sphere":
                r = float(g.get("radius"))
                v = box_vertices([2 * r] * 3)
            else:
                continue
            verts.append(v @ R.T + np.array(xyz))
        if verts:
            out[link.get("name")] = fit_capsule(np.vstack(verts))
    return out


class ObstacleCloudNode(Node):
    def __init__(self):
        super().__init__("pnp_obstacle_cloud")
        self.declare_parameter("rate_hz", 10.0)         # polls; a cloud is made once per detector frame (~5 Hz)
        self.declare_parameter("pixel_step", 4)          # every 4th pixel: 160x120 points instead of 640x480
        self.declare_parameter("voxel_m", 0.02)          # = Octomap resolution
        # points this close to the table top count as table (pnp_core/cloud.py): 15 mm hides a 11 mm brick
        self.declare_parameter("surface_band_m", 0.015)
        # the pick target in the map: "collision_object" = left in the cloud; the pick node adds it to MoveIt as an
        # object at its 6-D pose and MoveIt filters exactly that shape (an obstacle beside it stays whole).
        # "cylinder" = the old cut-out around each focus detection (it also erased what stood within ~2 cm).
        self.declare_parameter("target_cutout", "collision_object")
        self.declare_parameter("robot_margin_m", 0.03)    # beyond the collision geometry (TF / timing slack)
        self.declare_parameter("can_margin_m", 0.03)
        # Keep removing a can's points for this long after its last detection: while the arm reaches down, it
        # half-hides the can, YOLO can miss it for a few frames, and its still-visible top must not become an
        # obstacle right where the gripper is going.
        self.declare_parameter("can_hold_s", 5.0)
        # Real camera: stamp clouds with OUR clock. The frames carry the camera host's clock (e.g. a Jetson), and
        # MoveIt looks up the robot's TF at the cloud's stamp; a small clock offset would make that lookup fail.
        self.declare_parameter("restamp", False)
        # = max_range of the Octomap updater (config/octomap_sensors.yaml) + margin: removed points are pushed out
        # to here, so they only CLEAR space (see pnp_core.cloud.push_beyond)
        self.declare_parameter("clear_distance_m", 2.5)
        # EXPERIMENT knobs (docs/learning/09_sensor_view.md): deliberately mis-calibrate and watch the measured sizes
        # in RViz / tools/scene_check.py.   ros2 param set /pnp_obstacle_cloud experiment_fx_scale 1.05
        # (they also distort the Octomap the robot plans with: set them back to 1.0 before picking)
        self.declare_parameter("experiment_fx_scale", 1.0)       # focal length (fx AND fy) x this
        self.declare_parameter("experiment_depth_scale", 1.0)    # every depth value x this
        self.declare_parameter("scene", "config/scenes/basic.yaml")   # table + which object is the target
        # FROZEN MAP (pick & place map_mode: freeze_per_task, docs/learning/13_map_modes.md): while frozen, clouds are
        # NOT published (MoveIt's Octomap keeps the snapshot) but every frame is compared with the snapshot.
        # drop pixels at depth jumps > this (flying pixels of real stereo cameras: phantom obstacles beside the arm);
        # 0 = off. Their rays are lost for clearing too, which is fine: neighbors clear the same space.
        self.declare_parameter("edge_filter_m", 0.03)
        # a voxel is an obstacle only if this many depth points fall into it in one frame (noise speckle is single)
        self.declare_parameter("min_points_per_voxel", 3)
        self.declare_parameter("change_min_voxels", 20)     # smaller clusters are noise (a can is ~80 voxels)
        # the monitor ignores live voxels this close to the robot's links: at speed, a few arm points escape the
        # 3 cm map filter (harmless in the map) and looked like objects that "appeared" (8 false alarms in 20 tasks)
        self.declare_parameter("change_robot_margin_m", 0.08)
        self.declare_parameter("change_persist_frames", 3)   # a change must be seen this many frames in a row
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.p = p
        cam = load_camera()
        topics = load_yaml("config/interfaces.yaml")["topics"]
        scene = load_yaml(p("scene"))
        robot = load_robot()
        self.base, self.tcp = robot["frames"]["base"], robot["frames"]["tcp"]
        # the cell may tune the cloud (a desk cell with 1-2 cm objects: finer voxels, tighter margins)
        tuned = scene.get("perception", {}).get("obstacle_cloud", {})
        if tuned:
            self.set_parameters([rclpy.parameter.Parameter(k, value=type(self.get_parameter(k).value)(v))
                                 for k, v in tuned.items()])
            self.get_logger().info(f"scene tuning: {tuned}")
        self.optical = cam["frames"]["color_optical"]
        t = scene["table"]
        self.table_z = t["center"][2] + t["size"][2] / 2
        self.workspace = {"x": [t["center"][0] - t["size"][0] / 2, t["center"][0] + t["size"][0] / 2],
                          "y": [t["center"][1] - t["size"][1] / 2, t["center"][1] + t["size"][1] / 2], "z_max": 0.7}
        # the FOCUS object's points (the one being picked; std_msgs/String /perception/focus, default the scene
        # target) are removed in a vertical cylinder around each of its detections: radius = half the footprint (a
        # box: half its diagonal, so every yaw is covered), height = the object's. OTHER objects on the table stay in
        # the map: for this pick they are obstacles like any box.
        self.capsules = None                                  # {link: (a, b, r)} from /robot_description
        self._set_object(scene["target"]["object"])
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.focus_hint = None
        self.create_subscription(String, "/perception/focus", self._on_focus, latched)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.info, self.cans = None, []                       # cans: [(wall time seen, [x, y])]
        self.depths = deque(maxlen=15)                        # recent depth images (~1 s)
        self.det_stamp = None                                 # image stamp [ns] of the newest detection result
        self.capsules = None                                  # {link: (a, b, r)} from /robot_description
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/robot_description", self._urdf, latched)
        self.create_subscription(CameraInfo, topics["color_info"]["name"], lambda m: setattr(self, "info", m),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, topics["aligned_depth"]["name"], lambda m: self.depths.append(m),
                                 qos_profile_sensor_data)
        self.create_subscription(Detection3DArray, topics["detections"]["name"], self._dets, 10)
        self.pub = self.create_publisher(PointCloud2, "/perception/obstacle_cloud", 5)
        self.frozen, self.snapshot, self.recent = False, set(), deque(maxlen=3)   # recent: live voxel sets
        # the depth frames behind the snapshot (depth, T_cam_base, K): what the snapshot SAW as empty space
        self.recent_views, self.snap_views = deque(maxlen=3), []
        self.streak = {"appeared": 0, "gone": 0, "newly_seen": 0}
        self.create_service(SetBool, "/perception/map_freeze", self._freeze)
        self.pub_changes = self.create_publisher(String, "/perception/map_changes", 5)
        self.pub_change_mk = self.create_publisher(MarkerArray, "/perception/map_change_markers", 5)
        # where pick & place just put the object down: treat it as the target right away (before the detector has
        # seen it there), so a placed object never shows up as a new obstacle / a scene change
        self.create_subscription(String, "/pick_place/grasp", self._grasp_event,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        # the raw object points (base frame, before voxels/columns): input of scene_model_node (measuring)
        self.pub_objects = self.create_publisher(PointCloud2, "/perception/object_points", 5)
        # the TARGET cut-out made visible (RViz group 2): the shape around each focus detection, and the points it
        # removed -- whatever lies inside never reaches MoveIt, an obstacle right beside the target included
        self.pub_cut_mk = self.create_publisher(MarkerArray, "/perception/cutout_markers", 5)
        self.pub_cut_pts = self.create_publisher(PointCloud2, "/perception/cutout_points", 5)
        # everything standing on the table, NO target cut-out: what the pickability check tests the open jaws against
        self.pub_table_pts = self.create_publisher(PointCloud2, "/perception/table_points", 5)
        self.create_timer(1.0 / p("rate_hz"), self._tick)
        self.get_logger().info("obstacle cloud ready -> /perception/obstacle_cloud")

    def _on_focus(self, msg):
        name, self.focus_hint = parse_focus(msg.data)          # pnp_core/focus.py: + which instance (near here)
        self.cans = []                                        # the previous focus' cut-outs: obstacles again
        self._set_object(name)

    def _set_object(self, name):
        try:
            obj = load_object(name)
        except KeyError:
            self.get_logger().warn(f"focus '{name}': not in config/objects.yaml (kept the previous object)")
            return
        d = obj["dims_m"]
        self.focus_label = obj.get("yolo_class")
        self.can_r = d[0] / 2 if obj["shape"] == "cylinder" else float(np.hypot(d[0], d[1])) / 2
        self.can_h, self.grasp_depth = d[2], float(obj["grasp"]["depth_below_top_m"])
        if getattr(self, "tcp", None) and self.capsules and self.tcp in self.capsules:   # the carried-object capsule follows it
            top = self.grasp_depth
            self.capsules[self.tcp] = (np.array([0.0, 0.0, -top]), np.array([0.0, 0.0, self.can_h - top]),
                                       self.can_r + 0.01)
        self.get_logger().info(f"focus object: {name} ({self.focus_label})")

    def _urdf(self, msg):
        self.capsules = link_capsules(msg.data)
        # Plus the CAN while it's carried: it hangs below the TCP (tool z points out of the gripper). Without this its
        # points would turn into a pillar under it (MoveIt only excludes the attached can itself), right on the goal.
        # When nothing is held this space is empty air or the can about to be grasped, which must go anyway.
        top = self.grasp_depth                     # the TCP is this far below the object's top (objects.yaml)
        self.capsules[self.tcp] = (np.array([0.0, 0.0, -top]), np.array([0.0, 0.0, self.can_h - top]),
                                   self.can_r + 0.01)
        self.get_logger().info(f"robot self-filter: {len(self.capsules)} link capsules from /robot_description")

    def _freeze(self, req, resp):
        if req.data:
            self.snapshot = set().union(*self.recent) if self.recent else set()
            self.snap_views = list(self.recent_views)
            self.frozen, self.streak = True, {"appeared": 0, "gone": 0, "newly_seen": 0}
            resp.message = f"map frozen: snapshot of {len(self.snapshot)} occupied voxels"
        else:
            self.frozen, self.snapshot = False, set()
            resp.message = "map live (continuous updates)"
            self.pub_change_mk.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
        resp.success = True
        self.get_logger().info(resp.message)
        return resp

    def _grasp_event(self, msg):
        ev = json.loads(msg.data)
        if ev.get("state") == "released" and "T_base_obj" in ev:
            T = np.array(ev["T_base_obj"])
            self.cans.append((time.time(), [float(T[0, 3]), float(T[1, 3])]))

    def _monitor(self, live, depth, k, T_cam_base, header):
        """Frozen: compare this frame with the snapshot -> /perception/map_changes (JSON) + markers."""
        res, n_min = self.p("voxel_m"), self.p("change_min_voxels")
        def observed_free(keys):                     # seen through in any snapshot frame = was empty space then
            return set().union(*(seen_through(keys, res, d, kk[0], kk[4], kk[2], kk[5], tcb)
                                 for d, tcb, kk in self.snap_views)) if self.snap_views else keys
        app = appeared(live, self.snapshot, n_min, observed_free=observed_free)
        seen = newly_seen(live, self.snapshot, observed_free, n_min)
        gone = clusters(seen_through(self.snapshot, res, depth, k[0], k[4], k[2], k[5], T_cam_base), n_min)
        for name, comps in (("appeared", app), ("gone", gone), ("newly_seen", seen)):
            self.streak[name] = self.streak[name] + 1 if comps else 0
        persist = self.p("change_persist_frames")
        self.pub_changes.publish(String(data=json.dumps({
            "frozen": True, "snapshot_voxels": len(self.snapshot),
            "changed": max(self.streak["appeared"], self.streak["gone"]) >= persist,     # the scene changed
            "incomplete": self.streak["newly_seen"] >= persist,     # the snapshot missed something (was hidden)
            "appeared": summarize(app, res), "gone": summarize(gone, res), "newly_seen": summarize(seen, res)})))
        ma = MarkerArray()
        for i, (comps, rgba) in enumerate(((app, (1.0, 0.2, 0.2, 0.9)), (gone, (0.6, 0.6, 0.9, 0.6)),
                                           (seen, (1.0, 0.8, 0.2, 0.7)))):     # red / blue / amber
            m = Marker(ns="map_changes", id=i, type=Marker.CUBE_LIST, action=Marker.ADD)
            m.header.frame_id, m.header.stamp = self.base, header.stamp
            m.scale.x = m.scale.y = m.scale.z = res
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            m.points = [Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
                        for c in key_centers({kk for comp in comps for kk in comp}, res)]
            ma.markers.append(m)
        self.pub_change_mk.publish(ma)

    def _robot_capsules(self, stamp=None):
        caps = []
        for link, (a, b, r) in self.capsules.items():
            T = self._T(self.base, link, stamp)
            caps.append((T[:3, :3] @ a + T[:3, 3], T[:3, :3] @ b + T[:3, 3], r))
        return caps

    def _dets(self, msg):
        now = time.time()
        self.cans = [c for c in self.cans if now - c[0] < self.p("can_hold_s")]
        seen = [[d.bbox.center.position.x, d.bbox.center.position.y] for d in msg.detections
                if not d.results or d.results[0].hypothesis.class_id == self.focus_label]
        # only the focus INSTANCE (the hint: clicked / being picked) is cut out: another of its class is an obstacle
        self.cans += [(now, xy) for xy in nearest(seen, self.focus_hint)]
        self.det_stamp = Time.from_msg(msg.header.stamp).nanoseconds

    def _T(self, parent, child, stamp=None):
        if stamp is None:
            tf = self.tf.lookup_transform(parent, child, Time()).transform
        else:
            tf = self.tf.lookup_transform(parent, child, stamp, timeout=Duration(seconds=0.1)).transform
        q, t = tf.rotation, tf.translation
        return make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])

    def _tick(self):
        if self.det_stamp is None or self.info is None or self.capsules is None:
            return
        # Only use a depth image whose OWN detection result is already in (same image stamp). Otherwise, right after
        # the can moved, a frame would be processed before the detector reported the new position: the can becomes
        # a column, and the inside of that column is hidden behind the can's own surface, so no ray ever clears it
        # (seen: START_STATE_IN_COLLISION '<octomap> - grasped_object' at LIFT).
        match = [d for d in self.depths if abs(Time.from_msg(d.header.stamp).nanoseconds - self.det_stamp) < 5e7]
        if not match:
            return
        msg = match[-1]
        self.det_stamp = None
        try:
            T_base_cam = self._T(self.base, self.optical)
            # The robot where it was when the image was taken AND where it is now: a moving arm is in a different
            # place by the time we process the frame. With only "now", arm points from the image survive the filter
            # and become phantom columns along the arm's path (seen: START_STATE_IN_COLLISION at the can).
            caps = self._robot_capsules()
            try:
                caps += self._robot_capsules(Time.from_msg(msg.header.stamp))
            except Exception as e:  # noqa: BLE001  (no TF at that stamp yet / clock offset: "now" only)
                # without it a moving arm's old image position isn't filtered: phantom obstacles along its path
                self.get_logger().warn(f"no robot TF at the image's time, filtering with 'now' only: {str(e)[:120]}",
                                       throttle_duration_sec=5.0)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"TF not ready: {e}", throttle_duration_sec=5.0)
            return
        k = self.info.k
        depth = depth_to_meters(np.frombuffer(msg.data, dtype=np.uint16 if msg.encoding == "16UC1" else np.float32)
                                .reshape(msg.height, msg.width), msg.encoding)
        depth = drop_depth_edges(depth, self.p("edge_filter_m"))
        f_s = self.p("experiment_fx_scale")
        pts_cam = depth_to_cloud(depth * self.p("experiment_depth_scale"), k[0] * f_s, k[4] * f_s, k[2], k[5],
                                 step=self.p("pixel_step"))
        pts_base = pts_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
        cut, table = [], []
        cut_xy = [c[1] for c in self.cans if time.time() - c[0] < self.p("can_hold_s")]
        out, removed, objects = obstacle_cloud(pts_base, self.table_z, self.workspace, caps, self.p("robot_margin_m"),
                                      cut_xy,
                                      self.can_r + self.p("can_margin_m"), self.can_h, res=self.p("voxel_m"),
                                      surface_band=self.p("surface_band_m"), min_points=self.p("min_points_per_voxel"),
                                      cut_out=cut, cut_from_cloud=self.p("target_cutout") == "cylinder",
                                      table_points=table)
        T_cam_base = np.linalg.inv(T_base_cam)
        to_cam = lambda pts: pts @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]  # noqa: E731
        far = far_rays(depth * self.p("experiment_depth_scale"), k[0] * f_s, k[4] * f_s, k[2], k[5],
                       step=self.p("pixel_step"))        # beyond the mapping range: free space (pnp_core.cloud.far_rays)
        out_cam = np.vstack([to_cam(out), push_beyond(to_cam(removed), self.p("clear_distance_m")),
                             push_beyond(far, self.p("clear_distance_m"))])
        above = out[out[:, 2] > self.table_z + 0.02]                     # occupied voxels above the table
        if self.p("target_cutout") != "cylinder" and cut_xy:             # the target is in the cloud now: the change
            above = above[~in_cylinders(above, cut_xy, self.can_r + self.p("can_margin_m"),   # monitor ignores it
                                        self.table_z + self.can_h + 0.03)]
        live = voxel_keys(above, self.p("voxel_m"))
        self.recent.append(live)
        if not self.frozen:
            self.recent_views.append((depth, T_cam_base, tuple(k)))
        if self.frozen:                       # MoveIt keeps the snapshot; we only watch for changes
            near_robot = near_capsule_list(above, caps, self.p("change_robot_margin_m"))
            self._monitor(voxel_keys(above[~near_robot], self.p("voxel_m")), depth, k, T_cam_base, msg.header)
        else:
            self.pub_changes.publish(String(data=json.dumps({"frozen": False, "changed": False})))
            msg.header.frame_id = self.optical
            if self.p("restamp"):
                msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(point_cloud2.create_cloud_xyz32(msg.header, out_cam.astype(np.float32)))
        msg.header.frame_id = self.base
        self.pub_objects.publish(point_cloud2.create_cloud_xyz32(msg.header, objects.astype(np.float32)))
        table_pts = np.vstack(table) if table else np.zeros((0, 3))
        self.pub_table_pts.publish(point_cloud2.create_cloud_xyz32(msg.header, table_pts.astype(np.float32)))
        cut_pts = np.vstack(cut) if cut else np.zeros((0, 3))
        self.pub_cut_pts.publish(point_cloud2.create_cloud_xyz32(msg.header, cut_pts.astype(np.float32)))
        self.pub_cut_mk.publish(self._cutout_markers(msg.header, cut_xy if self.p("target_cutout") == "cylinder"
                                                     else []))

    def _cutout_markers(self, header, cut_xy):
        """One translucent cylinder per cut-out (pnp_core.cloud.obstacle_cloud: radius, top as it applies them)."""
        r = self.can_r + self.p("can_margin_m")
        h = self.can_h + 0.03
        ma = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, (x, y) in enumerate(cut_xy):
            m = Marker(ns="cutout", id=i, type=Marker.CYLINDER, action=Marker.ADD)
            m.header = header
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), self.table_z + h / 2
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * r
            m.scale.z = h
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.55, 0.0, 0.35
            ma.markers.append(m)
        return ma


def main():
    rclpy.init()
    node = ObstacleCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
