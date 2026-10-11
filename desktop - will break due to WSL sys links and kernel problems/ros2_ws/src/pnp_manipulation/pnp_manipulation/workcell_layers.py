"""What RViz shows, for the workcell page's 3D view: the robot model (its URDF + meshes, posed by /joint_states on the
page), every perception MarkerArray, the point clouds and the cameras' view cones, all in the robot base frame.

    GET /api/urdf              the robot description, mesh paths -> /api/mesh?f=... (the page builds the model)
    GET /api/mesh?f=<path>     one mesh file (only files under the repo / the workspace)
    GET /api/joints            {"t": stamp, "q": {joint: rad}}  (the page polls it ~20 Hz: the model follows the arm)
    GET /api/layers?want=a,b   {"list": [...], "layers": {id: {...}}}: markers / clouds / camera cones

Markers: kept the way RViz keeps them (per namespace + id; DELETE / DELETEALL / lifetime), moved into the base frame
by TF when asked. Clouds: subscribed only while a page shows them (asked < 3 s ago), downsampled to MAX_POINTS.
"""

import math
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray

from pnp_core.catalog import load_camera, repo_root
from pnp_core.geometry import quat_to_matrix
from pnp_core.wrist import wrist_camera

MAX_POINTS = 8000
MESH_TYPES = (".stl", ".dae", ".obj")
# id -> (label, kind, topic, colour for clouds without rgb)
LAYERS = {
    "detections": ("Detections", "markers", "/perception/markers", None),
    "pose": ("6-D pose", "markers", "/perception/object_pose_marker", None),
    "scene_model": ("Scene model", "markers", "/scene/model_markers", None),
    "map_change": ("Map changes", "markers", "/perception/map_change_markers", None),
    "cutouts": ("Cut-outs", "markers", "/perception/cutout_markers", None),
    "camera_rig": ("Camera rig", "markers", "/scene/camera_rig_markers", None),
    "cameras": ("Camera views", "frustum", None, None),
    "obstacle_cloud": ("Obstacle cloud", "cloud", "/perception/obstacle_cloud_filtered", [0.55, 0.6, 0.7]),
    "object_points": ("Object points", "cloud", "/perception/object_points", [0.18, 0.44, 0.93]),
    "cutout_points": ("Cut-out points", "cloud", "/perception/cutout_points", [0.95, 0.55, 0.15]),
    "depth_points": ("Overhead depth cloud", "cloud", "/camera/camera/depth/color/points", [0.7, 0.7, 0.7]),
}
LATCHED = {"/scene/camera_rig_markers"}


def _r(v, n=4):
    return [round(float(x), n) for x in v]


class Layers:
    def __init__(self, node, tf, base: str):
        self.node, self.tf, self.base = node, tf, base
        self.lock = threading.Lock()
        self.markers = {k: {} for k, v in LAYERS.items() if v[1] == "markers"}
        self.clouds, self.cloud_sub, self.cloud_asked, self.cloud_t = {}, {}, {}, {}
        self.urdf_raw = None
        self.q, self.q_t = {}, 0.0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        for lid, (_, kind, topic, _) in LAYERS.items():
            if kind == "markers":
                node.create_subscription(MarkerArray, topic, lambda m, lid=lid: self._on_markers(lid, m),
                                         latched if topic in LATCHED else 10)
        # the cameras' view cones: intrinsics + TF base -> optical
        self.cams = {"overhead": {"frame": load_camera()["frames"]["color_optical"], "info": None, "far": None}}
        node.create_subscription(CameraInfo, "/camera/camera/color/camera_info",
                                 lambda m: self.cams["overhead"].__setitem__("info", m), qos_profile_sensor_data)
        wc = wrist_camera(node.robot_cfg)
        if wc:
            self.cams["wrist"] = {"frame": wc["frames"]["color_optical"], "info": None, "far": 0.25}
            node.create_subscription(CameraInfo, wc["topics"]["color_info"],
                                     lambda m: self.cams["wrist"].__setitem__("info", m), qos_profile_sensor_data)
        node.create_timer(2.0, self._drop_idle_clouds)

    # --- robot model ------------------------------------------------------------------------------------------
    def on_urdf(self, xml: str):
        self.urdf_raw = xml

    def urdf(self) -> bytes | None:
        """The description with every mesh as a URL this server answers (file:///abs -> /api/mesh?f=/abs)."""
        if not self.urdf_raw:
            return None
        root = ET.fromstring(self.urdf_raw)
        for m in root.iter("mesh"):
            fn = m.get("filename", "")
            if fn.startswith("file://"):
                m.set("filename", "/api/mesh?f=" + fn[len("file://"):])
        for t in ("ros2_control", "transmission", "gazebo"):         # not needed by the page
            for e in root.findall(t):
                root.remove(e)
        return ET.tostring(root)

    @staticmethod
    def mesh(path: str) -> bytes | None:
        p = Path(path).resolve()
        roots = [repo_root().resolve(), Path("/ws").resolve(), Path("/opt/ros").resolve()]
        if p.suffix.lower() not in MESH_TYPES or not p.is_file() or not any(r in p.parents for r in roots):
            return None
        return p.read_bytes()

    def on_joints(self, msg):
        with self.lock:
            self.q.update(zip(msg.name, (float(v) for v in msg.position)))
            self.q_t = time.time()

    def joints(self):
        with self.lock:
            return {"t": self.q_t, "age": round(time.time() - self.q_t, 2) if self.q_t else None,
                    "q": {k: round(v, 5) for k, v in self.q.items()}}

    # --- markers ----------------------------------------------------------------------------------------------
    def _on_markers(self, lid, msg):
        now = time.time()
        with self.lock:
            store = self.markers[lid]
            for m in msg.markers:
                if m.action == Marker.DELETEALL:
                    store.clear()
                elif m.action == Marker.DELETE:
                    store.pop((m.ns, m.id), None)
                else:
                    store[(m.ns, m.id)] = (m, now)

    def _T(self, frame, stamp=None):
        """base <- frame (4x4) or None."""
        if not frame or frame == self.base:
            return np.eye(4)
        for t in ([stamp, Time()] if stamp is not None else [Time()]):
            try:
                tr = self.tf.lookup_transform(self.base, frame, t).transform
                T = np.eye(4)
                T[:3, :3] = quat_to_matrix([tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w])
                T[:3, 3] = [tr.translation.x, tr.translation.y, tr.translation.z]
                return T
            except Exception:  # noqa: BLE001
                continue
        return None

    def _marker_json(self, m):
        T = self._T(m.header.frame_id)
        if T is None:
            return None
        p, o = m.pose.position, m.pose.orientation
        P = np.eye(4)
        qn = math.sqrt(o.x ** 2 + o.y ** 2 + o.z ** 2 + o.w ** 2) or 1.0
        P[:3, :3] = quat_to_matrix([o.x / qn, o.y / qn, o.z / qn, o.w / qn])
        P[:3, 3] = [p.x, p.y, p.z]
        M = T @ P
        out = {"type": int(m.type), "ns": m.ns, "id": int(m.id), "m": _r(M[:3, :].T.ravel(), 5),
               "s": _r([m.scale.x, m.scale.y, m.scale.z]), "c": _r([m.color.r, m.color.g, m.color.b, m.color.a], 3)}
        if m.points:
            out["pts"] = _r([v for q in m.points for v in (q.x, q.y, q.z)])
        if m.colors:
            out["cols"] = _r([v for c in m.colors for v in (c.r, c.g, c.b)], 3)
        if m.text:
            out["text"] = m.text
        return out

    def _markers_out(self, lid):
        now = time.time()
        with self.lock:
            store = self.markers[lid]
            for k, (m, t) in list(store.items()):
                life = m.lifetime.sec + m.lifetime.nanosec * 1e-9
                if life > 0 and now - t > life:
                    del store[k]
            items = [m for m, _ in store.values()]
        return [j for j in (self._marker_json(m) for m in items) if j]

    # --- clouds -----------------------------------------------------------------------------------------------
    def _drop_idle_clouds(self):
        for lid in list(self.cloud_sub):
            if time.time() - self.cloud_asked.get(lid, 0) > 3.0:
                self.node.destroy_subscription(self.cloud_sub.pop(lid))
                self.clouds.pop(lid, None)

    def _ask_cloud(self, lid):
        self.cloud_asked[lid] = time.time()
        if lid not in self.cloud_sub:
            self.cloud_sub[lid] = self.node.create_subscription(PointCloud2, LAYERS[lid][2],
                                                                lambda m, lid=lid: self._on_cloud(lid, m),
                                                                qos_profile_sensor_data)

    def _on_cloud(self, lid, msg):
        if time.time() - self.cloud_t.get(lid, 0) < 0.5:             # <= 2 Hz is plenty for a page
            return
        self.cloud_t[lid] = time.time()
        names = [f.name for f in msg.fields]
        f = "rgb" if "rgb" in names else "rgba" if "rgba" in names else None
        try:
            arr = point_cloud2.read_points(msg, field_names=["x", "y", "z"] + ([f] if f else []), skip_nans=True)
            xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float64)
            rgb = None
            if f:
                u = np.ascontiguousarray(arr[f]).view(np.uint32)
                rgb = np.column_stack([(u >> 16) & 255, (u >> 8) & 255, u & 255]).astype(np.uint8)
        except Exception as e:  # noqa: BLE001
            self.node.get_logger().warn(f"cloud {LAYERS[lid][2]}: {e}", throttle_duration_sec=10.0)
            return
        if len(xyz) > MAX_POINTS:
            idx = np.linspace(0, len(xyz) - 1, MAX_POINTS).astype(int)
            xyz = xyz[idx]
            rgb = rgb[idx] if rgb is not None else None
        T = self._T(msg.header.frame_id, Time.from_msg(msg.header.stamp))
        if T is None:
            return
        xyz = xyz @ T[:3, :3].T + T[:3, 3]
        out = {"pts": np.round(xyz, 4).ravel().tolist(), "n": int(len(xyz)), "t": time.time(),
               "color": LAYERS[lid][3]}
        if rgb is not None and len(rgb) == len(xyz):
            out["cols"] = rgb.ravel().tolist()
        self.clouds[lid] = out

    # --- camera cones -----------------------------------------------------------------------------------------
    def _frustums(self, desk_z):
        out = []
        for name, c in self.cams.items():
            info = c["info"]
            T = self._T(c["frame"])
            if info is None or T is None:
                continue
            K = np.array(info.k).reshape(3, 3)
            w, h = info.width, info.height
            rays = np.array([[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]
                             for u, v in ((0, 0), (w, 0), (w, h), (0, h))]) @ T[:3, :3].T
            o = T[:3, 3]
            corners = []
            for r in rays:                                 # to the desk top, or `far` along the ray
                s = (desk_z - o[2]) / r[2] if r[2] < -1e-6 else None
                if c["far"] is not None or s is None or s > 3.0:
                    s = (c["far"] or 0.5)
                corners.append(_r(o + s * r))
            out.append({"name": name, "origin": _r(o), "corners": corners})
        return out

    # --- the page's query --------------------------------------------------------------------------------------
    def listing(self):
        return [{"id": k, "label": v[0], "kind": v[1], "topic": v[2]} for k, v in LAYERS.items()]

    def get(self, want, desk_z):
        out = {}
        for lid in want:
            if lid not in LAYERS:
                continue
            kind = LAYERS[lid][1]
            if kind == "markers":
                out[lid] = {"kind": kind, "markers": self._markers_out(lid)}
            elif kind == "cloud":
                self._ask_cloud(lid)
                c = self.clouds.get(lid)
                out[lid] = {"kind": kind, **(c or {"pts": [], "n": 0})}
            elif kind == "frustum":
                out[lid] = {"kind": kind, "cams": self._frustums(desk_z)}
        return {"list": self.listing(), "layers": out}
