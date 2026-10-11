"""The workcell: scan the table, show what the robot sees, pick an object, put it where you click.

    started by bringup (workcell:=true, the default in the sim)   ->  http://localhost:8088   (+ RViz: Workcell)

One node, three faces:
  web page   ui/workcell/index.html, served here: a top-down map + a 3D view of the WORLD MODEL (the table, the arm,
             the reach band, every detected object, the obstacles the camera measured) + the overhead camera
  RViz       the same objects as clickable markers (click = choose) and a goal disc you drag on the table, with a
             right-click menu "Pick & place" (interactive markers, topic /workcell)
  ROS        Scan = arm out of the camera's view, then ~1.5 s of detections averaged into a list of objects
             (/perception/detections: every class the detector knows). Pick = /pick_place with that object's
             class, the goal, and a hint "the one near here" (use_pick_hint) so the right instance is taken.
The world model is the SCAN (a snapshot you can look at and choose from), refreshed after every task.

HTTP API (JSON): GET /api/world, GET /api/camera.jpg, POST /api/scan | select {id} | goal {xy, yaw_deg} |
pick {id, xy, yaw_deg, rest?} | home | cancel | plan/add {id, xy, yaw_deg, rest?} | plan/update {index, xy?, yaw_deg?} |
plan/remove {index} | plan/clear | plan/run. Cameras: GET /api/feeds, GET /api/feed/<id>.jpg. The 3D view (the URDF
model on /joint_states + RViz's markers / clouds / camera cones): GET /api/urdf | mesh | joints | layers (workcell_layers.py).
"""

import json
import math
import threading
import time
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from interactive_markers import InteractiveMarkerServer, MenuHandler
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker

from pnp_core.brick import linear_to_srgb
from pnp_core.catalog import load_camera, load_robot, load_scene, load_yaml, repo_root
from pnp_core.focus import focus_message
from pnp_core.gripper import GripperModel
from pnp_core.object_grasp import upright_pose
from pnp_core.grasp import grasp_yaws
from pnp_core.pickability import check_inflate, pick_check
from pnp_core.geometry import make_transform, quat_to_matrix
from pnp_core.perception_math import depth_to_meters
from pnp_core.reorient import current_rest, plan_reorientation, rest_poses, transitions
from pnp_core.wrist import wrist_camera
from pnp_interfaces.action import PickPlace
from pnp_manipulation.robot_interface import RobotInterface
from pnp_manipulation.workcell_layers import Layers
from sensor_msgs.msg import JointState

UI = repo_root() / "ui" / "workcell" / "index.html"
DEFAULT_COLORS = {"soup_can": "#d0403a", "cracker_box": "#e3a23b"}


def hex_color(rgb_linear) -> str:
    c = (np.asarray(linear_to_srgb(rgb_linear)) * 255).round().astype(int)
    return "#%02x%02x%02x" % tuple(c)


class WorkcellNode(Node):
    def __init__(self):
        super().__init__("pnp_workcell")
        self.declare_parameter("port", 8088)
        self.declare_parameter("speed_profile", "dev")    # the PickPlace speed (robot arm.speed_profiles): dev | fast
        self.declare_parameter("scan_s", 1.5)
        self.scene, self.robot_cfg = load_scene(), load_robot()
        self.base = self.robot_cfg["frames"]["base"]
        self.catalog = load_yaml("config/objects.yaml")
        self.by_label = {o["yolo_class"]: (n, o) for n, o in self.catalog.items() if o.get("yolo_class")}
        self.lock = threading.Lock()
        self.objects, self.obstacles, self.selected, self.goal = [], [], None, None
        # which face each object rests on: what the robot did to it (class, xy, face), and the 6-D pose of the
        # object being tracked (/perception/pose_status); the detector alone can't tell
        self.rest_mem, self.pose_live, self.moves_cache = [], None, {}
        self.scan = {"time": None, "busy": False}
        self.task = {"busy": False, "state": None, "log": [], "result": None}
        self.goal_handle = None
        self.plan, self.plan_abort = [], False          # queued moves (plan_add), run in order by plan_run
        self.latest_dets, self.measured, self.known = [], [], []
        self.chain = []
        self.bridge = CvBridge()
        self.tf = Buffer()
        TransformListener(self.tf, self)
        topics = load_yaml("config/interfaces.yaml")["topics"]
        self.create_subscription(Detection3DArray, topics["detections"]["name"], self._on_dets, 10)
        # camera feeds for the page (GET /api/feed/<id>.jpg): subscribed only while a page shows them (images are
        # big; six feeds at 15 Hz would cost the container bandwidth + JPEG encoding for nothing)
        self.feeds = {"detections": ("Overhead: detections", "/perception/debug_image", "color"),
                      "overhead": ("Overhead: color", topics["color_image"]["name"], "color"),
                      "overhead_depth": ("Overhead: depth", topics["aligned_depth"]["name"], "depth"),
                      "mask": ("Detector mask (focus object)", "/perception/mask", "labels")}
        wc = wrist_camera(self.robot_cfg)
        if wc:
            self.feeds["wrist"] = ("Wrist: color", wc["topics"]["color_image"], "color")
            self.feeds["wrist_depth"] = ("Wrist: depth", wc["topics"]["aligned_depth"], "depth")
        if wc:
            self.feeds["wrist_map"] = ("Wrist map (top view)", "/wrist_map/topview", "map")   # latched, one per scan
        self.feed_msg, self.feed_sub, self.feed_asked, self.feed_got, self.feed_jpg = {}, {}, {}, {}, {}
        self.wrist_scan_cli = self.create_client(Trigger, "/wrist_map/scan")
        self.wrist_scan = {"busy": False, "result": None}
        self.feed_lock = threading.Lock()
        self.create_timer(2.0, self._drop_idle_feeds)
        # the camera's view of the desk: a goal must be inside it, or the placement can't be checked (seen: a goal
        # behind the robot was accepted; the arm put the can down out of view, next to its own base)
        self.cam_info, self.view = None, None
        self.cam_frame = load_camera()["frames"]["color_optical"]
        self.create_subscription(CameraInfo, topics["color_info"]["name"],
                                 lambda m: setattr(self, "cam_info", m), qos_profile_sensor_data)
        self.create_subscription(String, "/scene/model", self._on_model, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/scene/known_obstacles", self._on_known,          # the sim: volatile,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))   # re-sent every second
        self.create_subscription(String, "/robot_description", self._on_urdf, latched)
        self.create_subscription(String, "/perception/pose_status", self._on_pose_status, 10)
        # the CLICKED object becomes perception's focus (pnp_core/focus.py): FoundationPose switches to it and
        # registers on that instance, the obstacle map carves only it out. pick_place_node publishes the same topic
        # when a pick starts (it owns the focus while a task runs).
        self.pub_focus = self.create_publisher(String, "/perception/focus", latched)
        self.placing = None          # while a task runs: the object being carried there (RViz goal: orange + ghost)
        # pickable? after each scan, the open hand's space at every grasp vs everything on the table (no target
        # cut-out; pnp_core/pickability.py, the pick node's check before it moves): blocked objects are greyed out
        self.table_points = None
        self.create_subscription(PointCloud2, "/perception/table_points", self._on_table_points, 2)
        # ... and the selected object again, once a second, at the 6-D pose the pick will grasp from (the scan's pose
        # is the detector's: a few mm / degrees off is enough to flip a tight case)
        self.create_timer(1.0, self._recheck_selected)
        self.pp = ActionClient(self, PickPlace, "/pick_place")
        self.R = RobotInterface(self, self.robot_cfg)
        # the 3D view's robot model + RViz's layers (markers, clouds, camera cones): pnp_manipulation/workcell_layers.py
        self.layers = Layers(self, self.tf, self.base)
        if getattr(self, "_urdf_early", None):
            self.layers.on_urdf(self._urdf_early)
        self.create_subscription(JointState, "/joint_states", self.layers.on_joints, 20)
        # RViz: clickable objects + a goal disc to drag
        self.im = InteractiveMarkerServer(self, "workcell")
        self.goal_menu = MenuHandler()
        self.goal_menu.insert("Pick & place the chosen object here", callback=lambda fb: self._pick_from_rviz())
        self.get_logger().info(f"workcell page: http://localhost:{self.get_parameter('port').value}")

    # --- inputs ---------------------------------------------------------------------------------------------------
    def _on_dets(self, msg):
        dets = []
        for d in msg.detections:
            p, q = d.bbox.center.position, d.bbox.center.orientation
            dets.append({"class": d.results[0].hypothesis.class_id if d.results else "?",
                         "score": float(d.results[0].hypothesis.score) if d.results else 0.0,
                         "xyz": [p.x, p.y, p.z], "yaw": 2 * math.atan2(q.z, q.w),
                         "size": [d.bbox.size.x, d.bbox.size.y, d.bbox.size.z]})
        with self.lock:
            self.latest_dets.append((time.time(), dets))
            self.latest_dets = [x for x in self.latest_dets if time.time() - x[0] < 5.0]

    # --- camera feeds ---------------------------------------------------------------------------------------------
    def feed_list(self):
        now = time.time()
        return [{"id": k, "label": v[0], "topic": v[1], "kind": v[2],
                 "age_s": round(now - self.feed_got[k], 1) if k in self.feed_got else None}
                for k, v in self.feeds.items()]

    def _drop_idle_feeds(self):
        with self.feed_lock:
            for k in [k for k, t in self.feed_asked.items() if time.time() - t > 10.0 and k in self.feed_sub]:
                self.destroy_subscription(self.feed_sub.pop(k))
                self.feed_msg.pop(k, None)

    def feed_jpeg(self, fid):
        """The newest frame of feed `fid` as JPEG (subscribes on first use), or None (unknown / nothing yet)."""
        if fid not in self.feeds:
            return None
        with self.feed_lock:
            self.feed_asked[fid] = time.time()
            if fid not in self.feed_sub:
                def store(m, fid=fid):
                    self.feed_msg[fid], self.feed_got[fid] = m, time.time()
                qos = qos_profile_sensor_data if self.feeds[fid][2] != "map" else QoSProfile(
                    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
                self.feed_sub[fid] = self.create_subscription(Image, self.feeds[fid][1], store, qos)
            msg = self.feed_msg.get(fid)
            cached = self.feed_jpg.get(fid)
        if msg is None:
            return None
        if cached and cached[0] is msg:
            return cached[1]
        img = self._render_feed(msg, self.feeds[fid][2])
        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        with self.feed_lock:
            self.feed_jpg[fid] = (msg, buf.tobytes())
        return buf.tobytes()

    def _render_feed(self, msg, kind):
        if kind in ("color", "map"):
            return self.bridge.imgmsg_to_cv2(msg, "bgr8")
        raw = self.bridge.imgmsg_to_cv2(msg, "passthrough")
        if kind == "labels":                         # 0 = background, 1.. = detections (instance labels)
            lab = raw.astype(np.int32) if raw.ndim == 2 else raw[..., 0].astype(np.int32)
            pal = np.array([[30, 30, 30], [80, 200, 60], [60, 140, 255], [255, 120, 60], [200, 80, 220], [60, 220, 220]],
                           np.uint8)
            return pal[np.where(lab > 0, (lab - 1) % (len(pal) - 1) + 1, 0)]
        # depth: turbo colors (near = red) over the nearest 2-90 % of the pixels; farther saturates blue. Over 2-98 %
        # the overhead camera's floor (3.7 m) took most of the scale and the desk + objects were one flat color
        d = depth_to_meters(raw, msg.encoding).astype(np.float32)
        ok = np.isfinite(d) & (d > 0)
        out = np.zeros(d.shape + (3,), np.uint8)
        if ok.any():
            lo, hi = np.percentile(d[ok], [2, 90])
            n = np.clip((d - lo) / max(hi - lo, 1e-3), 0, 1)
            col = cv2.applyColorMap((255 * (1 - n)).astype(np.uint8), cv2.COLORMAP_TURBO)    # near = red
            out[ok] = col[ok]
            cv2.putText(out, f"{lo:.2f}-{hi:.2f} m", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    def _on_pose_status(self, msg):
        d = json.loads(msg.data)
        if d.get("state") in ("TRACKING", "HOLD") and d.get("T_base_obj"):
            self.pose_live = (d.get("object"), np.array(d["T_base_obj"]), time.time(), d.get("state"))

    # --- resting faces ---------------------------------------------------------------------------------------------
    def _moves(self, name):
        if name not in self.moves_cache:
            self.moves_cache[name] = transitions(self.catalog[name], self.robot_cfg["gripper"])
        return self.moves_cache[name]

    def _rest_of(self, name, cls, xy, height=None):
        """The face an object at xy rests on: the live 6-D pose if it is this object, else what we did to it, else
        the face whose height matches what the detector measured (+ / - of a face look alike: the + one)."""
        if self.pose_live and self.pose_live[0] == name and time.time() - self.pose_live[2] < 3.0:
            T = self.pose_live[1]
            if math.hypot(T[0, 3] - xy[0], T[1, 3] - xy[1]) < 0.04:
                return current_rest(T[:3, :3], self.catalog[name]) or "upright"
        for m in reversed(self.rest_mem):
            if m["class"] == cls and math.hypot(m["xy"][0] - xy[0], m["xy"][1] - xy[1]) < 0.04:
                return m["rest"]
        if height and name in self.catalog and "shape" in self.catalog[name]:
            faces = rest_poses(self.catalog[name])
            best = min(faces.values(), key=lambda r: abs(2 * r.height_m - height))
            if abs(2 * faces["upright"].height_m - height) > abs(2 * best.height_m - height) + 0.01:
                return best.name
        return "upright"

    def _remember_rest(self, cls, xy, rest):
        self.rest_mem = [m for m in self.rest_mem if not (m["class"] == cls and
                                                          math.hypot(m["xy"][0] - xy[0], m["xy"][1] - xy[1]) < 0.04)]
        self.rest_mem.append({"class": cls, "xy": list(xy), "rest": rest})

    def _faces(self, o):
        """Fill an object entry's faces: each face's look, and how many moves this robot needs to get it there."""
        name = o["name"]
        if name not in self.catalog or "shape" not in self.catalog[name]:
            o.update(rests=[], rest="upright")
            return o
        cur = o.get("rest") or "upright"
        moves = self._moves(name)
        o["rests"] = []
        for r in rest_poses(self.catalog[name]).values():
            path = plan_reorientation(self.catalog[name], self.robot_cfg["gripper"], cur, r.name, moves)
            o["rests"].append({"name": r.name, "label": r.label, "footprint": [round(v, 4) for v in r.footprint_m],
                               "height": round(r.height_m, 4), "period_deg": r.period_deg,
                               "moves": None if path is None else len(path),
                               "via": [] if not path else [m.to for m in path]})
        return o

    def _footprint_half(self, o, rest=None):
        r = next((f for f in o.get("rests", []) if f["name"] == (rest or o.get("rest"))), None)
        return math.hypot(*(r["footprint"] if r else o["dims"][:2])) / 2

    def _on_model(self, msg):
        self.measured = json.loads(msg.data).get("measured", [])

    def _on_known(self, msg):
        self.known = json.loads(msg.data).get("boxes", [])

    def _on_urdf(self, msg):
        """The arm's chain base -> TCP (link names), for drawing it from TF; the page's 3D model gets the whole URDF."""
        if hasattr(self, "layers"):
            self.layers.on_urdf(msg.data)
        else:
            self._urdf_early = msg.data
        root = ET.fromstring(msg.data)
        parent = {j.find("child").get("link"): j.find("parent").get("link") for j in root.findall("joint")}
        link, chain = self.robot_cfg["frames"]["tcp"], []
        while link and link != self.base:
            chain.append(link)
            link = parent.get(link)
        self.chain = [self.base] + chain[::-1]

    # --- goal checks ----------------------------------------------------------------------------------------------
    def _camera_view(self):
        """The part of the desk the overhead camera sees: the image corners (5 % in) projected onto the desk top,
        as a polygon [[x, y], ...] in the base frame. None until camera_info + TF are there."""
        if self.view is not None or self.cam_info is None:
            return self.view
        try:
            tf = self.tf.lookup_transform(self.base, self.cam_frame, Time()).transform
        except Exception:  # noqa: BLE001
            return None
        T = make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                           [tf.translation.x, tf.translation.y, tf.translation.z])
        K = np.array(self.cam_info.k, float).reshape(3, 3)
        w, h, m = self.cam_info.width, self.cam_info.height, 0.05
        t = self.scene["table"]
        top = t["center"][2] + t["size"][2] / 2
        poly = []
        for u, v in [(m * w, m * h), ((1 - m) * w, m * h), ((1 - m) * w, (1 - m) * h), (m * w, (1 - m) * h)]:
            ray = T[:3, :3] @ np.linalg.solve(K, [u, v, 1.0])
            if ray[2] >= -1e-6:                        # a corner looking at or above the horizon: no polygon
                return None
            k = (top - T[2, 3]) / ray[2]
            poly.append([round(float(T[0, 3] + k * ray[0]), 4), round(float(T[1, 3] + k * ray[1]), 4)])
        self.view = poly                               # the camera is fixed: compute once
        return poly

    @staticmethod
    def _inside(poly, x, y):
        inside = False
        for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside

    def goal_problem(self, xy, obj=None):
        """Why a goal can't be used (None: fine): not a number, the object would hang off the desk, outside the
        reach band, or where the camera can't see it (the placement is verified with the camera)."""
        try:
            x, y = float(xy[0]), float(xy[1])
        except (TypeError, ValueError, IndexError):
            return "the goal must be two numbers [x, y] in meters"
        if not (math.isfinite(x) and math.isfinite(y)):
            return "the goal must be finite"
        t = self.scene["table"]
        r = self._footprint_half(obj, obj.get("goal_rest")) if obj else 0.0
        if abs(x - t["center"][0]) > t["size"][0] / 2 - r or abs(y - t["center"][1]) > t["size"][1] / 2 - r:
            return "off the desk" + (f" (the {obj['label']} would hang over the edge)" if obj else "")
        reach = self.scene.get("ui", {}).get("reach_m")
        if reach and not reach[0] <= math.hypot(x, y) <= reach[1]:
            return f"outside the reach band ({reach[0]:.2f}-{reach[1]:.2f} m from the base)"
        view = self._camera_view()
        if view is not None and not self._inside(view, x, y):
            return "outside the camera's view (the placement couldn't be checked)"
        return None

    @staticmethod
    def _blocked_by(xy, obj, objects, moved_away=()):
        """The label of another object the chosen one would land on (footprint circles + 1 cm), skipping objects
        listed in moved_away ([class, xy] of objects an earlier plan step takes away), else None."""
        half = lambda o: (math.hypot(*next((f["footprint"] for f in o.get("rests", []) if f["name"] ==   # noqa: E731
                                              (o.get("goal_rest") or o.get("rest"))), o["dims"][:2])) / 2)
        for o in objects:
            if obj and o.get("id") is not None and o["id"] == obj.get("id"):
                continue
            if any(c == o["class"] and math.hypot(o["xy"][0] - m[0], o["xy"][1] - m[1]) < 0.03 for c, m in moved_away):
                continue
            if math.hypot(o["xy"][0] - xy[0], o["xy"][1] - xy[1]) < half(o) + (half(obj) if obj else 0.0) + 0.01:
                return o["label"]
        return None

    # --- the world model ------------------------------------------------------------------------------------------
    def _arm_points(self):
        pts = []
        for link in self.chain:
            try:
                t = self.tf.lookup_transform(self.base, link, Time()).transform.translation
            except Exception:  # noqa: BLE001
                continue
            if not pts or np.linalg.norm(np.subtract([t.x, t.y, t.z], pts[-1])) > 0.01:
                pts.append([round(t.x, 4), round(t.y, 4), round(t.z, 4)])
        return pts

    def _object_entry(self, k, cls, xyz, yaw, score, size, n):
        name, o = self.by_label.get(cls, (cls, {"shape": "box", "dims_m": size}))
        color = hex_color(o["color"]) if "color" in o else DEFAULT_COLORS.get(cls, "#7d8ca8")
        label = cls.replace("_", " ")
        g = o.get("grasp", {})
        period = None if g.get("symmetric_yaw") else float(g.get("yaw_period_deg", 360))   # None: round, any yaw
        return {"id": f"o{k}", "class": cls, "name": name, "label": label, "xy": [round(xyz[0], 4), round(xyz[1], 4)],
                "z_center": round(xyz[2], 4), "yaw": round(yaw, 4), "dims": list(o.get("dims_m", size)),
                "shape": o.get("shape", "box"), "color": color, "score": round(score, 3), "seen": n,
                "yaw_period_deg": period, "measured_height": round(float(size[2]), 4) if size else None}

    def first_scan(self, timeout_s=180.0):
        """The first scan, once the detector reports objects: at bringup YOLO and the sim's camera take a while
        (a fixed 4 s wait scanned an empty table). Retries while the scan finds nothing."""
        end = time.time() + timeout_s
        while time.time() < end and rclpy.ok():
            with self.lock:
                seen = any(dets for t, dets in self.latest_dets if t > time.time() - 2.0)
            if seen and not self.scan["busy"]:
                self._scan()
                if self.objects:
                    self.get_logger().info(f"first scan: {len(self.objects)} objects")
                    return
            time.sleep(2.0)
        self.get_logger().warn("first scan: no objects seen; press Scan on the page")

    def _scan(self):
        """Arm out of the way, then average what the detector sees for scan_s seconds into a list of objects."""
        with self.lock:
            self.scan["busy"] = True
        try:
            if self.R.wait_ready(timeout=5.0):
                self.R.move_joints(self.robot_cfg["arm"]["home"])
            time.sleep(0.8)
            t0 = time.time()
            time.sleep(float(self.get_parameter("scan_s").value))
            with self.lock:
                frames = [d for t, d in self.latest_dets if t >= t0]
            clusters = []                                        # [class, [xyz...], [yaw...], [score...], size]
            for dets in frames:
                for d in dets:
                    for c in clusters:
                        if c[0] == d["class"] and np.hypot(*(np.mean(c[1], axis=0)[:2] - d["xyz"][:2])) < 0.03:
                            c[1].append(d["xyz"]), c[2].append(d["yaw"]), c[3].append(d["score"])
                            break
                    else:
                        clusters.append([d["class"], [d["xyz"]], [d["yaw"]], [d["score"]], d["size"]])
            need = max(1, len(frames) // 3)                      # seen in at least a third of the frames
            objs = []
            for c in sorted((c for c in clusters if len(c[1]) >= need), key=lambda c: -np.mean(c[3])):
                xyz = np.mean(c[1], axis=0)
                yaw = float(np.angle(np.mean(np.exp(2j * np.array(c[2])))) / 2)     # box yaw: 180-deg symmetric
                objs.append(self._object_entry(len(objs) + 1, c[0], xyz.tolist(), yaw, float(np.mean(c[3])), c[4],
                                               len(c[1])))
            for o in objs:
                o["rest"] = self._rest_of(o["name"], o["class"], o["xy"], o.get("measured_height"))
                self._faces(o)
            self._pickability(objs)
            with self.lock:
                old = next((o for o in self.objects if o["id"] == self.selected), None)
                self.objects = objs
                # keep the choice if the same object is still there
                self.selected = next((o["id"] for o in objs if old and o["class"] == old["class"]
                                      and np.hypot(o["xy"][0] - old["xy"][0], o["xy"][1] - old["xy"][1]) < 0.05), None)
                self.obstacles = self._obstacles(objs)
                self.scan["time"] = time.time()
            self._update_rviz()
            self._publish_focus()                       # the kept selection, at where the scan now sees it
        finally:
            with self.lock:
                self.scan["busy"] = False

    def _obstacles(self, objs):
        """Boxes the camera measured (scene model), minus the ones that are detected objects; the sim's known
        layout when there's no measurement."""
        boxes = self.measured or self.known
        out = []
        for b in boxes:
            c = b["center"]
            if any(np.hypot(c[0] - o["xy"][0], c[1] - o["xy"][1]) < max(o["dims"][:2]) / 2 + 0.02 for o in objs):
                continue
            out.append({"center": [round(v, 4) for v in c], "size": [round(v, 4) for v in b["size"]],
                        "yaw": round(float(b.get("yaw", 0.0)), 4)})
        return out

    def world(self):
        t, ui = self.scene["table"], self.scene.get("ui", {})
        reach = ui.get("reach_m")
        with self.lock:
            w = {"now": time.time(), "cell": {"robot": self.robot_cfg["name"], "scene": self.scene.get("name", "?")},
                 "desk": {"center": t["center"], "size": t["size"]},
                 "areas": {"pick": self.scene["target"]["spawn_region"], "place": self.scene["goal"]["region"]},
                 "reach": {"r_min": reach[0], "r_max": reach[1]} if reach else None,
                 "objects": self.objects, "obstacles": self.obstacles, "selected": self.selected,
                 "goal": self.goal, "scan": dict(self.scan), "task": {**self.task, "log": self.task["log"][-60:]},
                 "plan": [dict(st) for st in self.plan], "wrist_scan": dict(self.wrist_scan)}
        w["robot"] = {"links": self._arm_points(), "base": self.base}
        w["camera_view"] = self._camera_view()
        if self.goal and reach:
            r = math.hypot(*self.goal["xy"])
            w["goal_reachable"] = reach[0] <= r <= reach[1]
        if self.goal:
            w["goal_problem"] = self.goal_problem(self.goal["xy"],
                                                  next((o for o in w["objects"] if o["id"] == w["selected"]), None))
        if self.goal:                                            # would the chosen object land on another one?
            sel = next((o for o in w["objects"] if o["id"] == w["selected"]), None)
            w["goal_blocked_by"] = self._blocked_by(self.goal["xy"], sel, w["objects"]) if sel else None
        return w

    # --- actions --------------------------------------------------------------------------------------------------
    def start_scan(self):
        if self.task["busy"] or self.scan["busy"]:
            return {"ok": False, "why": "busy"}
        threading.Thread(target=self._scan, daemon=True).start()
        return {"ok": True}

    def _on_table_points(self, msg):
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.table_points = (time.time(), np.asarray(pts, float).reshape(-1, 3))

    def _pickability(self, objs):
        """o["pickable"] True / False (+ o["blocked"]: why) / None (not checked: no fresh table points, or the object
        rests on another face) -- the same check the pick node runs before it moves (pnp_core/pickability.py: same
        grasp yaws, same inflate, and for the selected object the same 6-D pose)."""
        for o in objs:
            self._pick_check(o)

    def _live_pose_of(self, o):
        """The pose estimator's TRACKING pose if it is this object (same catalog name, within 3 cm), fresh -> T, else
        None. What the pick grasps from (pick_place_node._object_pose, grasp_source auto)."""
        p = self.pose_live
        if p is None or len(p) < 4 or p[3] != "TRACKING" or p[0] != o["name"] or time.time() - p[2] > 2.0:
            return None
        T = p[1]
        return T if math.hypot(T[0, 3] - o["xy"][0], T[1, 3] - o["xy"][1]) < 0.03 else None

    def _pick_check(self, o):
        tp = self.table_points
        o["pickable"], o["blocked"], o["checked_with"] = None, None, None
        obj = self.catalog.get(o["name"])
        if tp is None or time.time() - tp[0] > 3.0 or obj is None or (o.get("rest") or "upright") != "upright":
            return
        T = self._live_pose_of(o)
        src = "6-D pose" if T is not None else "scan"
        if T is None:
            T = upright_pose(obj, o["xy"], o["z_center"], o["yaw"])
        g = self.robot_cfg["gripper"]
        sym = obj.get("grasp", {}).get("symmetric_yaw", False)
        try:
            r = pick_check(tp[1], obj, T, g, GripperModel(g), self.R.tcp_yaw(), grasp_yaws(0.0, self.R.tcp_yaw(), sym),
                           approach_height=float(self.robot_cfg.get("grasp_motion", {}).get("approach_height_m", 0.10)),
                           label=o["label"], inflate=check_inflate(self.scene),
                           flips=bool(self.scene.get("grasping", {}).get("scoring", False)))   # = the pick's
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"pickability of {o['label']}: {e}")
            return
        o["pickable"], o["blocked"], o["checked_with"] = bool(r["pickable"]), (r["why"] or None), src

    def _recheck_selected(self):
        if self.task["busy"] or self.scan["busy"]:
            return
        with self.lock:
            o = next((x for x in self.objects if x["id"] == self.selected), None)
        if o is None or self._live_pose_of(o) is None:
            return
        o2 = dict(o)
        self._pick_check(o2)
        with self.lock:
            if o2["checked_with"] is not None and any(x is o for x in self.objects):
                o.update(pickable=o2["pickable"], blocked=o2["blocked"], checked_with=o2["checked_with"])

    def select(self, oid):
        with self.lock:
            self.selected = oid if any(o["id"] == oid for o in self.objects) else None
        self._update_rviz()
        self._publish_focus()
        return {"ok": self.selected is not None}

    def _publish_focus(self):
        """The selected object -> /perception/focus (its catalog name + where it is: "this one"). Not while a task
        runs: the pick owns the focus then (switching the pose estimator mid-pick would lose the object)."""
        with self.lock:
            o = next((o for o in self.objects if o["id"] == self.selected), None)
        if o is None or self.task["busy"]:
            return
        msg = focus_message(o["name"], o["xy"])     # re-sending is harmless: the same spot doesn't re-register
        self.pub_focus.publish(String(data=msg))
        self.get_logger().info(f"focus -> {msg}")

    def set_goal(self, xy, yaw_deg=0.0, from_rviz=False, rest=None):
        why = self.goal_problem(xy)
        if why and why.startswith(("the goal", "off the desk")):
            return {"ok": False, "why": why}
        yaw_deg = float(yaw_deg)
        if not math.isfinite(yaw_deg):
            return {"ok": False, "why": "the yaw must be finite"}
        with self.lock:
            if rest is None and self.goal:
                rest = self.goal.get("rest") if not from_rviz else self.goal.get("rest")
            self.goal = {"xy": [round(float(xy[0]), 4), round(float(xy[1]), 4)], "yaw_deg": float(yaw_deg),
                         "rest": rest or None}
        if not from_rviz:
            self._update_rviz()
        return {"ok": True}

    def _face_path(self, obj, rest):
        """-> (faces to put it on, one per move; error). [None] = one move keeping its face."""
        cur = obj.get("rest") or "upright"
        if not rest or rest == cur:
            return [None], None
        if obj["name"] not in self.catalog or rest not in rest_poses(self.catalog[obj["name"]]):
            return None, f"the {obj['label']} has no face '{rest}'"
        path = plan_reorientation(self.catalog[obj["name"]], self.robot_cfg["gripper"], cur, rest,
                                  self._moves(obj["name"]))
        if path is None:
            lab = rest_poses(self.catalog[obj["name"]])[rest].label
            return None, (f"the {self.robot_cfg['name']} gripper can't turn the {obj['label']} {lab}: "
                          f"no grasp of it can do that (tools/reorient_graph.py)")
        return [m.to for m in path], None

    def pick(self, oid=None, xy=None, yaw_deg=None, rest=None, preview=False):
        with self.lock:
            oid = oid or self.selected
            obj = next((o for o in self.objects if o["id"] == oid), None)
            goal = self.goal if xy is None else {"xy": xy, "yaw_deg": yaw_deg or 0.0, "rest": rest}
            if self.task["busy"] or self.scan["busy"] or obj is None or goal is None:
                return {"ok": False, "why": "busy" if self.task["busy"] or self.scan["busy"]
                        else "choose an object and a goal first"}
            faces, err = self._face_path(obj, goal.get("rest"))
            if err:
                return {"ok": False, "why": err}
            probe = {**obj, "goal_rest": faces[-1]}
            why = self.goal_problem(goal["xy"], probe)
            if why:
                return {"ok": False, "why": f"goal {why}"}
            on = self._blocked_by(goal["xy"], probe, self.objects)
            if on:            # the pick node would refuse too, but only after detecting + planning (~20 s)
                return {"ok": False, "why": f"the goal is on the {on}: move it first (a plan can)"}
            goal = {"xy": [float(goal["xy"][0]), float(goal["xy"][1])], "yaw_deg": float(goal.get("yaw_deg") or 0.0)}
            how = "" if faces == [None] else " turning it " + " -> ".join(
                rest_poses(self.catalog[obj["name"]])[f].label for f in faces) + f" ({len(faces)} moves)"
            self.task = {"busy": True, "state": "starting", "result": None,
                         "log": [f"{'PREVIEW (plan only) of ' if preview else ''}pick {obj['label']} at "
                                 f"({obj['xy'][0]:.3f}, {obj['xy'][1]:.3f}) -> "
                                 f"({goal['xy'][0]:.3f}, {goal['xy'][1]:.3f}), {goal['yaw_deg']:.0f} deg{how}"]}
        if not preview:
            self.placing = obj
        self._update_rviz()
        threading.Thread(target=self._run_pick, args=(obj, goal, faces, bool(preview)), daemon=True).start()
        return {"ok": True}

    def _run_pick(self, obj, goal, faces=(None,), preview=False):
        src, line = obj["xy"], ""
        if preview:                      # the first move only, nothing moves, the world is unchanged: no rescan
            ok, line = self._execute(obj["class"], src, {**goal, "rest": faces[0]}, preview=True)
            with self.lock:
                self.task.update(busy=False, state=None, result=line)
                self.task["log"].append(line)
            return
        for k, face in enumerate(faces):
            if len(faces) > 1:
                with self.lock:
                    self.task["log"].append(f"move {k + 1}/{len(faces)}: "
                                            f"{rest_poses(self.catalog[obj['name']])[face].label}")
            ok, line = self._execute(obj["class"], src, {**goal, "rest": face})
            if not ok:
                break
            if face:
                self._remember_rest(obj["class"], goal["xy"], face)
            src = goal["xy"]
        with self.lock:
            self.task.update(busy=False, state=None, result=line)
            self.task["log"].append(line)
        self.placing = None
        self._update_rviz()
        self._scan()                                             # the world changed: look again

    def _execute(self, cls, from_xy, goal, preview=False):
        """/pick_place for the object of class `cls` near from_xy -> goal {xy, yaw_deg}; blocks. -> (success, line)"""
        ok = False
        try:
            if not self.pp.wait_for_server(timeout_sec=10.0):
                raise RuntimeError("/pick_place is not running")
            g = PickPlace.Goal(object_class=cls, use_default_place=False, place_x=float(goal["xy"][0]),
                               place_y=float(goal["xy"][1]), place_yaw=math.radians(goal["yaw_deg"]),
                               speed_profile=self.get_parameter("speed_profile").value, use_pick_hint=True,
                               pick_x=float(from_xy[0]),
                               pick_y=float(from_xy[1]), place_rest=goal.get("rest") or "", preview=bool(preview))

            def on_fb(fb):
                f = fb.feedback
                with self.lock:
                    self.task["state"] = f.state
                    self.task["log"].append(f"[{f.state}] {f.detail}" if f.detail else f"[{f.state}]")
            fut = self.pp.send_goal_async(g, feedback_callback=on_fb)
            while not fut.done():
                time.sleep(0.05)
            self.goal_handle = fut.result()
            if not self.goal_handle.accepted:
                raise RuntimeError("goal rejected")
            rf = self.goal_handle.get_result_async()
            while not rf.done():
                time.sleep(0.1)
            res = rf.result().result
            ok = res.outcome == PickPlace.Result.SUCCESS
            line = f"{res.outcome_name}: {res.message}"
            if res.placement_error_m == res.placement_error_m:
                line += f" ({res.placement_error_m * 1000:.1f} mm from the goal)"
        except Exception as e:  # noqa: BLE001
            line = f"ERROR: {e}"
        self.goal_handle = None
        return ok, line

    # --- a plan: several moves, run in order ------------------------------------------------------------------------
    def plan_add(self, oid=None, xy=None, yaw_deg=0.0, rest=None):
        """Queue "move this object there". An object an earlier step already moves is picked from THAT step's goal
        (chained moves), so the plan can be made from one scan."""
        with self.lock:
            obj = next((o for o in self.objects if o["id"] == (oid or self.selected)), None)
            xy = xy or (self.goal or {}).get("xy")
            if obj is None or xy is None:
                return {"ok": False, "why": "choose an object and a goal first"}
            if self.task["busy"]:
                return {"ok": False, "why": "busy: add moves before Run plan"}
            # the face it will be on when this step starts: an earlier step may already turn it
            cur = obj.get("rest") or "upright"
            for st in self.plan:
                if st["status"] == "queued" and st["class"] == obj["class"] and st.get("rest"):
                    cur = st["rest"]
            faces, err = self._face_path({**obj, "rest": cur}, rest)
            if err:
                return {"ok": False, "why": err}
            probe = {**obj, "goal_rest": faces[-1] or cur}
            why = self.goal_problem(xy, probe)
            if why:
                return {"ok": False, "why": f"goal {why}"}
            queued = [st for st in self.plan if st["status"] == "queued"]
            on = self._blocked_by(xy, probe, self.objects, [(st["class"], st["from"]) for st in queued])
            if on:
                return {"ok": False, "why": f"the goal is on the {on} (no earlier move takes it away)"}
            src = list(obj["xy"])
            for st in self.plan:
                if st["status"] == "queued" and st["class"] == obj["class"] and math.hypot(st["from"][0] - src[0], st["from"][1] - src[1]) < 0.03:
                    src = list(st["to"])
            to = [round(float(xy[0]), 4), round(float(xy[1]), 4)]
            labels = {r.name: r.label for r in rest_poses(self.catalog[obj["name"]]).values()} \
                if obj["name"] in self.catalog and "shape" in self.catalog[obj["name"]] else {}
            for face in faces:              # one step per move (a face change can take several, all at the goal)
                rinfo = next((f for f in obj.get("rests", []) if f["name"] == (face or cur)), None)
                self.plan.append({"class": obj["class"], "label": obj["label"], "color": obj["color"],
                                  "shape": obj["shape"], "dims": obj["dims"], "rests": obj.get("rests", []),
                                  "yaw_period_deg": (rinfo or {}).get("period_deg", obj.get("yaw_period_deg")),
                                  "footprint": (rinfo or {}).get("footprint"), "rest": face,
                                  "rest_label": labels.get(face) if face else None,
                                  "from": src, "to": to, "yaw_deg": float(yaw_deg if yaw_deg is not None else 0.0),
                                  "status": "queued", "result": None})
                src = to
        return {"ok": True, "steps": len(self.plan)}

    def plan_update(self, index, xy=None, yaw_deg=None):
        """Change a queued step's goal and / or yaw. A later step that starts where this one ended (chained) follows."""
        with self.lock:
            i = int(index)
            if self.task["busy"] or not 0 <= i < len(self.plan) or self.plan[i]["status"] != "queued":
                return {"ok": False, "why": "busy" if self.task["busy"] else "no such queued step"}
            st = self.plan[i]
            if yaw_deg is not None:
                y = float(yaw_deg)
                if not math.isfinite(y):
                    return {"ok": False, "why": "the yaw must be finite"}
                st["yaw_deg"] = y
            if xy is not None:
                me = {"id": None, "label": st["label"], "dims": st.get("footprint") or st["dims"]}
                why = self.goal_problem(xy, me)
                if why:
                    return {"ok": False, "why": f"goal {why}"}
                earlier = [(e["class"], e["from"]) for e in self.plan[:i] if e["status"] == "queued"]
                mine = [o for o in self.objects if not (o["class"] == st["class"] and
                                                        math.hypot(o["xy"][0] - st["from"][0], o["xy"][1] - st["from"][1]) < 0.03)]
                on = self._blocked_by(xy, me, mine, earlier)
                if on:
                    return {"ok": False, "why": f"the goal is on the {on} (no earlier move takes it away)"}
                old, new = st["to"], [round(float(xy[0]), 4), round(float(xy[1]), 4)]
                st["to"] = new
                for later in self.plan[i + 1:]:
                    if later["class"] == st["class"] and math.hypot(later["from"][0] - old[0], later["from"][1] - old[1]) < 0.03:
                        later["from"] = list(new)
                        break
        return {"ok": True}

    def plan_remove(self, index):
        with self.lock:
            if self.task["busy"] or not 0 <= int(index) < len(self.plan):
                return {"ok": False}
            self.plan.pop(int(index))
        return {"ok": True}

    def plan_clear(self):
        with self.lock:
            if self.task["busy"]:
                return {"ok": False, "why": "busy"}
            self.plan = []
        return {"ok": True}

    def plan_run(self):
        with self.lock:
            todo = [st for st in self.plan if st["status"] != "done"]
            if self.task["busy"] or self.scan["busy"] or not todo:
                return {"ok": False, "why": "busy" if self.task["busy"] else "nothing to run"}
            for st in todo:
                st.update(status="queued", result=None)
            self.plan_abort = False
            self.task = {"busy": True, "state": "plan", "result": None, "log": [f"plan: {len(todo)} move(s)"]}
        threading.Thread(target=self._run_plan, daemon=True).start()
        return {"ok": True}

    def _run_plan(self):
        done = 0
        steps = [st for st in self.plan if st["status"] == "queued"]
        for k, st in enumerate(steps):
            if self.plan_abort:
                break
            with self.lock:
                st["status"] = "running"
                self.task["log"].append(f"step {k + 1}/{len(steps)}: {st['label']} ({st['from'][0]:.3f}, "
                                        f"{st['from'][1]:.3f}) -> ({st['to'][0]:.3f}, {st['to'][1]:.3f}), "
                                        f"{st['yaw_deg']:.0f} deg")
            ok, line = self._execute(st["class"], st["from"], {"xy": st["to"], "yaw_deg": st["yaw_deg"],
                                                               "rest": st.get("rest")})
            if ok and st.get("rest"):
                self._remember_rest(st["class"], st["to"], st["rest"])
            with self.lock:
                st.update(status="done" if ok else "failed", result=line)
                self.task["log"].append(f"step {k + 1}: {line}")
            done += ok
            self._scan()
            if not ok:
                break
        with self.lock:
            for st in self.plan:
                if st["status"] in ("queued", "running"):
                    st["status"] = "skipped"
            result = f"{'SUCCESS' if done == len(steps) else 'STOPPED'}: plan {done}/{len(steps)} moves done"
            self.task.update(busy=False, state=None, result=result)
            self.task["log"].append(result)

    def start_wrist_scan(self):
        """The desk flat map from the wrist camera (wrist_map_node): the arm moves, so not while a task runs."""
        if self.task["busy"] or self.scan["busy"] or self.wrist_scan["busy"]:
            return {"ok": False, "why": "busy"}
        if not self.wrist_scan_cli.service_is_ready():
            return {"ok": False, "why": "no /wrist_map/scan (wrist_map_node not running?)"}
        self.wrist_scan.update(busy=True, result=None)

        def done(fut):
            r = fut.result()
            self.wrist_scan.update(busy=False, result=(r.message if r else "no answer"))
        self.wrist_scan_cli.call_async(Trigger.Request()).add_done_callback(done)
        return {"ok": True}

    def home(self):
        if self.task["busy"]:
            return {"ok": False, "why": "busy"}
        threading.Thread(target=lambda: self.R.move_joints(self.robot_cfg["arm"]["home"]), daemon=True).start()
        return {"ok": True}

    def cancel(self):
        self.plan_abort = True                                   # a running plan stops after this move
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            return {"ok": True}
        return {"ok": False}

    # --- RViz ------------------------------------------------------------------------------------------------------
    def _update_rviz(self):
        # re-insert (an update) instead of clear() + insert: a cleared marker gets a fresh context, and the server then
        # rejects RViz's feedback on it for 1 s ("conflicting clients"): a drag right after a click was lost
        ids = {o["id"] for o in self.objects}
        for name in list(self.im.marker_contexts):
            if name != "goal" and name not in ids:                 # objects gone since the last scan
                self.im.erase(name)
        top = self.scene["table"]["center"][2] + self.scene["table"]["size"][2] / 2
        for o in self.objects:
            m = InteractiveMarker(name=o["id"], scale=0.15, description=f"{o['label']} {int(o['score'] * 100)}%")
            m.header.frame_id = self.base
            m.pose.position.x, m.pose.position.y, m.pose.position.z = o["xy"][0], o["xy"][1], o["z_center"]
            m.pose.orientation.z, m.pose.orientation.w = math.sin(o["yaw"] / 2), math.cos(o["yaw"] / 2)
            shape = Marker(type=Marker.CYLINDER if o["shape"] == "cylinder" else Marker.CUBE)
            d = o["dims"]
            shape.scale.x, shape.scale.y, shape.scale.z = d[0] * 1.06, (d[0] if o["shape"] == "cylinder" else d[1]) * 1.06, d[2]
            sel = o["id"] == self.selected
            c = [int(o["color"][i:i + 2], 16) / 255 for i in (1, 3, 5)]
            shape.color.r, shape.color.g, shape.color.b, shape.color.a = (0.2, 0.45, 1.0, 0.75) if sel else (*c, 0.45)
            ctl = InteractiveMarkerControl(interaction_mode=InteractiveMarkerControl.BUTTON, always_visible=True)
            ctl.markers.append(shape)
            m.controls.append(ctl)
            self.im.insert(m, feedback_callback=lambda fb, oid=o["id"]: self._rviz_click(fb, oid))
        # the goal's state at a glance: grey = none set (it sits at the scene's default spot), green = set,
        # orange + a ghost of the object = a task is carrying it there now
        state = "placing" if self.placing is not None and self.goal else ("set" if self.goal else "unset")
        col = {"unset": (0.55, 0.55, 0.55, 0.55), "set": (0.12, 0.62, 0.33, 0.9), "placing": (1.0, 0.55, 0.0, 0.95)}[state]
        desc = {"unset": "goal: none set -- drag me", "set": "goal: drag me, right-click to pick",
                "placing": f"placing the {self.placing['label']} here" if self.placing else ""}[state]
        g = InteractiveMarker(name="goal", scale=0.12, description=desc)
        g.header.frame_id = self.base
        xy = self.goal["xy"] if self.goal else self.scene["goal"]["xy"]
        g.pose.position.x, g.pose.position.y, g.pose.position.z = xy[0], xy[1], top + 0.003
        yaw = math.radians(self.goal["yaw_deg"]) if self.goal else 0.0
        g.pose.orientation.z, g.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        disc = Marker(type=Marker.CYLINDER)
        disc.scale.x = disc.scale.y = 0.05 * max(self.scene["table"]["size"][:2]) / 1.3   # FR3 table: 5 cm
        disc.scale.z = 0.004
        disc.color.r, disc.color.g, disc.color.b, disc.color.a = col
        move = InteractiveMarkerControl(interaction_mode=InteractiveMarkerControl.MOVE_PLANE, always_visible=True)
        move.orientation.w, move.orientation.y = math.sqrt(0.5), math.sqrt(0.5)     # the plane normal = z: the table
        move.markers.append(disc)
        arrow = Marker(type=Marker.ARROW)             # the goal's yaw: the object's x axis there
        arrow.scale.x, arrow.scale.y, arrow.scale.z = disc.scale.x * 1.2, disc.scale.x * 0.12, disc.scale.x * 0.12
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = (*col[:3], 1.0)
        move.markers.append(arrow)
        if state == "placing":                         # a ghost of the object where it will land
            d, o = self.placing["dims"], self.placing
            ghost = Marker(type=Marker.CYLINDER if o["shape"] == "cylinder" else Marker.CUBE)
            ghost.scale.x, ghost.scale.y = d[0], (d[0] if o["shape"] == "cylinder" else d[1])
            ghost.scale.z = d[2]
            ghost.pose.position.z = d[2] / 2 - 0.003
            ghost.pose.orientation.w = 1.0
            ghost.color.r, ghost.color.g, ghost.color.b, ghost.color.a = 1.0, 0.55, 0.0, 0.35
            move.markers.append(ghost)
        g.controls.append(move)
        turn = InteractiveMarkerControl(name="turn", interaction_mode=InteractiveMarkerControl.ROTATE_AXIS)
        turn.orientation.w, turn.orientation.y = math.sqrt(0.5), math.sqrt(0.5)     # about z
        g.controls.append(turn)
        self.im.insert(g, feedback_callback=self._rviz_goal)
        self.goal_menu.apply(self.im, "goal")
        self.im.applyChanges()

    def _rviz_click(self, fb, oid):
        if fb.event_type == fb.BUTTON_CLICK:
            self.select(oid)

    def _rviz_goal(self, fb):
        if fb.event_type == fb.MOUSE_UP:
            p, q = fb.pose.position, fb.pose.orientation
            yaw = math.degrees(2 * math.atan2(q.z, q.w))           # the ring turns it about z
            self.set_goal([p.x, p.y], (yaw + 180) % 360 - 180, from_rviz=True)

    def _pick_from_rviz(self):
        self.get_logger().info(f"RViz: pick {self.selected} -> {self.goal}")
        r = self.pick()
        if not r.get("ok"):
            self.get_logger().warn(f"RViz pick refused: {r.get('why')}")


def make_handler(node: WorkcellNode):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):            # quiet
            pass

        def _send(self, code, body, ctype="application/json", cache="no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, Path(UI).read_bytes(), "text/html; charset=utf-8")
            elif path == "/grasp_study":                 # tools/grasp_study.py's report
                f = repo_root() / "data" / "grasp_study" / "index.html"
                self._send(200, f.read_bytes() if f.exists() else
                           b"<p>No grasp study yet: python3 /repo/tools/grasp_study.py --all</p>", "text/html; charset=utf-8")
            elif path == "/api/world":
                self._send(200, json.dumps(node.world()).encode())
            elif path == "/api/joints":
                self._send(200, json.dumps(node.layers.joints()).encode())
            elif path == "/api/urdf":
                x = node.layers.urdf()
                self._send(200 if x else 404, x or b"<robot/>", "application/xml")
            elif path == "/api/mesh":
                f = parse_qs(urlparse(self.path).query).get("f", [""])[0]
                data = node.layers.mesh(f)
                if data is None:
                    self._send(404, b"{}")
                else:
                    self._send(200, data, "application/octet-stream", "max-age=86400")
            elif path == "/api/layers":
                want = [w for w in parse_qs(urlparse(self.path).query).get("want", [""])[0].split(",") if w]
                t = node.scene["table"]
                self._send(200, json.dumps(node.layers.get(want, t["center"][2] + t["size"][2] / 2)).encode())
            elif path == "/api/feeds":
                self._send(200, json.dumps(node.feed_list()).encode())
            elif path == "/api/camera.jpg" or (path.startswith("/api/feed/") and path.endswith(".jpg")):
                jpg = node.feed_jpeg("detections" if path == "/api/camera.jpg" else path[len("/api/feed/"):-4])
                if jpg is None:
                    self._send(404, b"{}")
                else:
                    self._send(200, jpg, "image/jpeg")
            else:
                self._send(404, b"{}")

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("the body must be a JSON object")
            except ValueError as e:
                return self._send(400, json.dumps({"ok": False, "why": f"bad request: {e}"}).encode())
            path = self.path.split("?")[0]
            routes = {"/api/scan": lambda: node.start_scan(),
                      "/api/select": lambda: node.select(body.get("id")),
                      "/api/goal": lambda: node.set_goal(body["xy"], body.get("yaw_deg", 0.0), rest=body.get("rest")),
                      "/api/pick": lambda: node.pick(body.get("id"), body.get("xy"), body.get("yaw_deg"),
                                                     body.get("rest"), bool(body.get("preview", False))),
                      "/api/home": lambda: node.home(),
                      "/api/wrist_scan": lambda: node.start_wrist_scan(),
                      "/api/plan/add": lambda: node.plan_add(body.get("id"), body.get("xy"), body.get("yaw_deg"),
                                                             body.get("rest")),
                      "/api/plan/remove": lambda: node.plan_remove(body.get("index", -1)),
                      "/api/plan/update": lambda: node.plan_update(body.get("index", -1), body.get("xy"),
                                                                   body.get("yaw_deg")),
                      "/api/plan/clear": lambda: node.plan_clear(),
                      "/api/plan/run": lambda: node.plan_run(),
                      "/api/cancel": lambda: node.cancel()}
            if path not in routes:
                return self._send(404, b"{}")
            try:
                self._send(200, json.dumps(routes[path]()).encode())
            except (KeyError, IndexError, TypeError, ValueError) as e:     # wrong / missing fields
                self._send(400, json.dumps({"ok": False, "why": f"bad request: {e!r}"}).encode())
            except Exception as e:  # noqa: BLE001
                self._send(500, json.dumps({"ok": False, "why": str(e)}).encode())
    return Handler


def main():
    rclpy.init()
    node = WorkcellNode()
    httpd = ThreadingHTTPServer(("0.0.0.0", int(node.get_parameter("port").value)), make_handler(node))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=node.first_scan, daemon=True).start()   # a first look once the detector sees something
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    httpd.shutdown()
    node.destroy_node()
    rclpy.try_shutdown()
