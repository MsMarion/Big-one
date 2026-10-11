"""6-D pose of the target object: FoundationPose registers it once, then tracks it every frame.

    started by bringup.launch.py pose_estimation:=true  (needs the pnp:pose image; runs in /opt/fp_venv)

Inputs : color + aligned depth + camera_info + /perception/mask (detector: label image, same stamp as the frame)
Outputs: /perception/object_pose         geometry_msgs/PoseStamped, base frame (fr3_link0): the object's mesh frame
         /perception/object_pose_marker  visualization_msgs/MarkerArray: the textured mesh at the pose + axes + state
         /perception/pose_status         std_msgs/String JSON: state, timings, counters (for tools and the docs)
Control: /perception/pose_reset          std_srvs/Trigger: forget the pose, register again

States
  SEARCHING  no target in the mask yet
  TRACKING   each new frame: refine from the last pose (iterations.track, default 2), ~tens of ms
  MISMATCH   the detector sees the target, but not where the pose is (it was moved, or tracking slid off): the pose
             is probably stale; after `reregister_after` such frames -> register again
  HOLD       target hidden (no mask where the pose is, e.g. the gripper is above it): keep the last pose, don't track
             (tracking without the object in view can slide onto whatever covers it)
  ATTACHED   the gripper holds it (/pick_place/grasp "attached" + grasp offset T_tcp_obj): pose = TCP pose (TF) x
             T_tcp_obj, every tick, no images needed (the hand covers it; the camera would only see fingers)
Registration (iterations.register = 5 refine passes on 252 guesses, ~1.5 s) happens when: the first target appears;
the tracked pose and the detector disagree for `reregister_after` frames (pose center outside the target's mask);
the target reappears after HOLD somewhere else; or on /perception/pose_reset.
After "released" (with where it was put down) the node goes to HOLD at that pose: as soon as the hand is out of the
way it TRACKS from there (2 refine passes) instead of registering again. "lost" (dropped) = SEARCHING.

Close look (/perception/refine_pose, pnp_interfaces/RefinePose; docs/learning/14_wrist_camera.md): pick & place
calls it at the pre-grasp pose. The node takes a WRIST camera (D405) frame taken after the request, moves the
current pose estimate into the wrist camera's frame (TF at the image's time) and refines it there with
FoundationPose's tracker (no mask needed: tracking starts from a pose). The close view (~13 cm) sees the object
much larger than the fixed camera at ~1 m. The result goes back into the fixed-camera tracking.

Handing a pose to FoundationPose: its tracker continues from est.pose_last (in the frame of the camera it is
given, "per the centered mesh"); every handover (release, close look) must set it, not just our own copy.
"""

import math
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from pnp_interfaces.srv import RefinePose
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from pnp_core.focus import parse_focus  # noqa: E402
from pnp_core.catalog import load_robot, load_scene, load_yaml
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix
from pnp_core.wrist import wrist_camera

FP_CFG = load_yaml("config/foundationpose.yaml")
REPO = Path(os.environ.get("PNP_REPO", "/repo"))


def img_to_np(m):
    dt = {"rgb8": np.uint8, "bgr8": np.uint8, "mono8": np.uint8, "16UC1": np.uint16, "32FC1": np.float32}[m.encoding]
    a = np.frombuffer(m.data, dt).reshape(m.height, m.width, -1).squeeze()
    if m.encoding == "bgr8":
        a = a[..., ::-1]
    if m.encoding == "16UC1":
        a = a.astype(np.float32) / 1000.0
    return np.ascontiguousarray(a)


class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__("pnp_pose_estimator")
        self.declare_parameter("object", load_yaml("config/scenes/basic.yaml")["target"]["object"])  # launch: scene's
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("reregister_after", 3)        # frames of disagreement with the detector
        self.declare_parameter("mask_dilate_px", 6)
        # The pose must also agree with the detector's 3D estimate of the same detection (xy) within this. The mask
        # test alone is weak for big objects: a 16 cm box moved by 8 cm can still cover the old pose's center, and
        # tracking then continues from the wrong place (seen: 85 mm off). Detector bias: ~1.5 mm can, ~8 mm box.
        self.declare_parameter("max_offset_m", 0.03)
        self.declare_parameter("refine_iterations", 6)       # close look: refine passes on the wrist frame
        # only wrist frames taken this long AFTER the request: the arm has just stopped, and a frame from while it was
        # still settling, paired with the TF of the stopped arm, shifts the pose by that motion
        self.declare_parameter("refine_settle_s", 0.4)
        p = lambda n: self.get_parameter(n).value  # noqa: E731
        self.p = p
        self.obj = p("object")
        frames = load_robot()["frames"]
        self.base, self.tcp = frames["base"], frames["tcp"]
        topics = load_yaml("config/interfaces.yaml")["topics"]

        from pnp_perception.foundationpose_onnx import make_estimator   # heavy: GPU, TensorRT engines
        t = time.time()
        self.est = make_estimator(self.obj)
        self.mesh_path = str(REPO / FP_CFG["meshes"][self.obj])
        self.get_logger().info(f"FoundationPose ready for '{self.obj}' in {time.time() - t:.1f} s "
                               f"(engine: {self.est.refiner.model.engine})")
        # Every object of the scene (target + extra_objects) prepared NOW: FoundationPose's reset_object (mesh
        # diameter from 10k points, downsampling, GPU tensors) took ~9 s when done at the moment of a pick; a switch
        # is then a swap of the prepared state (_on_focus).
        self.obj_states = {self.obj: self._snapshot()}
        scene = load_scene()
        for name in [e["object"] for e in scene.get("extra_objects", [])]:
            if name in FP_CFG["meshes"] and name not in self.obj_states:
                t = time.time()
                self._reset_to(name)
                self.obj_states[name] = self._snapshot()
                self.get_logger().info(f"prepared '{name}' in {time.time() - t:.1f} s")
        self._restore(self.obj)

        self.tf = Buffer()
        TransformListener(self.tf, self)
        # The detector's mask arrives ~0.2-0.3 s after its camera frame (YOLO time). Pairing the NEWEST depth with
        # the newest mask almost never matched (tracking ran every few seconds instead of every frame), so keep ~2 s
        # of frames and pair each new mask with the color/depth it was computed from (same stamp).
        self.rgbs, self.depths = deque(maxlen=40), deque(maxlen=40)
        self.info, self.mask = None, None
        self.create_subscription(Image, topics["color_image"]["name"], lambda m: self.rgbs.append(m), qos_profile_sensor_data)
        self.create_subscription(Image, topics["aligned_depth"]["name"], lambda m: self.depths.append(m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, topics["color_info"]["name"], lambda m: setattr(self, "info", m), qos_profile_sensor_data)
        self.create_subscription(Image, "/perception/mask", lambda m: setattr(self, "mask", m), qos_profile_sensor_data)
        self.dets = {}                # image stamp [ns] -> detection centers (base frame), detection k = mask label k+1
        self.create_subscription(Detection3DArray, topics["detections"]["name"], self._on_dets, 10)
        self.pub_pose = self.create_publisher(PoseStamped, "/perception/object_pose", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/perception/object_pose_marker", 10)
        self.pub_status = self.create_publisher(String, "/perception/pose_status", 10)
        self.create_service(Trigger, "/perception/pose_reset", self._reset)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/pick_place/grasp", self._on_grasp, latched)
        # which object to estimate (several on the table: the one being picked; detector_node masks only that one)
        self.create_subscription(String, "/perception/focus", self._on_focus, latched)
        # wrist camera (config/robot_fr3.yaml wrist_camera): buffers + the close-look service. The service runs in its
        # own thread (it waits for a fresh wrist frame), so the estimator is shared under a lock.
        self.est_lock = threading.Lock()
        self.wrist = None
        self.wrist = wrist_camera(load_robot())
        if self.wrist:
            wt, cb = self.wrist["topics"], ReentrantCallbackGroup()
            self.w_rgbs, self.w_depths, self.w_info = deque(maxlen=10), deque(maxlen=10), None
            self.create_subscription(Image, wt["color_image"], lambda m: self.w_rgbs.append(m), qos_profile_sensor_data,
                                     callback_group=cb)
            self.create_subscription(Image, wt["aligned_depth"], lambda m: self.w_depths.append(m),
                                     qos_profile_sensor_data, callback_group=cb)
            self.create_subscription(CameraInfo, wt["color_info"], lambda m: setattr(self, "w_info", m),
                                     qos_profile_sensor_data, callback_group=cb)
            self.create_service(RefinePose, "/perception/refine_pose", self._refine,
                                callback_group=MutuallyExclusiveCallbackGroup())

        self.state, self.pose_cam, self.last_stamp = "SEARCHING", None, None
        self.cam_frame, self.T_tcp_obj, self.was, self.last_emit = None, None, None, None
        self.min_stamp = 0          # [ns] ignore camera frames older than this (taken before the last grasp event)
        self.disagree, self.reason = 0, "waiting for the target"
        self.stats = {"registrations": 0, "tracked_frames": 0, "last_register_ms": None, "last_track_ms": None}
        self.create_timer(1.0 / p("rate_hz"), self._tick)

    # --- inputs / control ---------------------------------------------------------------------------------------
    def _reset(self, _req, resp):
        self.state, self.pose_cam, self.reason = "SEARCHING", None, "reset requested"
        resp.success, resp.message = True, "pose forgotten; will register on the next frame with the target"
        return resp

    STATE_ATTRS = ("model_center", "mesh_ori", "diameter", "vox_size", "dist_bin", "angle_bin", "max_xyz", "min_xyz",
                   "pts", "normals", "mesh_path", "mesh", "mesh_tensors", "symmetry_tfs")

    def _snapshot(self):
        return {a: getattr(self.est, a) for a in self.STATE_ATTRS if hasattr(self.est, a)}

    def _restore(self, name):
        for a, v in self.obj_states[name].items():
            setattr(self.est, a, v)

    def _reset_to(self, name):
        import trimesh
        mesh = trimesh.load(str(REPO / FP_CFG["meshes"][name]), force="mesh")
        self.est.reset_object(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh)

    def _on_focus(self, msg):
        """Switch to another object: FoundationPose keeps its networks, only the object state changes (prepared at
        startup, so a swap of a few attributes), then it registers afresh on the new object's mask."""
        name, hint = parse_focus(msg.data)                    # pnp_core/focus.py
        old_hint, self.focus_hint = getattr(self, "focus_hint", None), hint
        if name and name == self.obj and hint is not None and (old_hint is None or math.dist(hint, old_hint) > 0.02):
            # the SAME class, another instance clicked (or the first hint): register afresh on the hinted mask
            with self.est_lock:
                self.est.pose_last = None
                self.min_stamp = self.get_clock().now().nanoseconds + int(0.2e9)
                self.state, self.pose_cam, self.T_tcp_obj = "SEARCHING", None, None
                self.reason = f"'{name}' near ({hint[0]:.3f}, {hint[1]:.3f}): registering on it"
            self.get_logger().info(f"focus -> '{name}' near ({hint[0]:.3f}, {hint[1]:.3f}): re-registering")
            return
        if not name or name == self.obj:
            return
        if name not in FP_CFG["meshes"]:
            self.get_logger().warn(f"focus '{name}': no FoundationPose mesh (config/foundationpose.yaml): kept '{self.obj}'")
            return
        mesh_path = str(REPO / FP_CFG["meshes"][name])
        t = time.time()
        with self.est_lock:
            if name in self.obj_states:              # prepared at startup: a swap
                self._restore(name)
            else:
                self._reset_to(name)
                self.obj_states[name] = self._snapshot()
            self.est.pose_last = None
            self.obj, self.mesh_path = name, mesh_path
            # the detector's masks show only the focus object; masks still buffered from before ITS switch show the
            # OLD one, and registering the new mesh on one gave a pose of the previous object (seen: 151 mm off)
            self.min_stamp = self.get_clock().now().nanoseconds + int(0.2e9)
            self.state, self.pose_cam, self.T_tcp_obj = "SEARCHING", None, None
            self.reason = f"switched to '{name}': registering on it"
        self.get_logger().info(f"focus -> '{name}' in {(time.time() - t) * 1000:.0f} ms (mesh {mesh_path})")

    def _on_grasp(self, msg):
        """What the gripper does with the object (pick_place_node): attached / released / lost."""
        ev = json.loads(msg.data)
        if ev.get("object") not in (None, self.obj):
            return
        s = ev["state"]
        # Frames from BEFORE this event show the old situation (e.g. the mask from before the carry is still in the
        # buffer after the release: the object is no longer there). Seen: one pose 448 mm off and a needless
        # re-registration right after a release.
        self.min_stamp = self.get_clock().now().nanoseconds
        if s == "attached":
            self.T_tcp_obj = np.array(ev["T_tcp_obj"])
            self.state, self.reason = "ATTACHED", "in the gripper: pose = TCP pose x grasp offset"
        elif s == "released":
            self.T_tcp_obj = None
            T_base_cam = self._T_base_cam(self.cam_frame) if self.cam_frame else None
            if T_base_cam is None:
                self.state, self.pose_cam, self.reason = "SEARCHING", None, "released, camera pose unknown"
            else:
                with self.est_lock:
                    self._handover(np.linalg.inv(T_base_cam) @ np.array(ev["T_base_obj"]))
                self.state, self.reason, self.disagree = "HOLD", "released: expecting it where it was put down", 0
        elif s == "lost":
            self.T_tcp_obj = None
            self.state, self.pose_cam, self.reason = "SEARCHING", None, "dropped / lost: looking for it again"
        self.get_logger().info(f"grasp event '{s}' -> {self.state}")

    def _handover(self, pose_in_cam, fixed_camera=True):
        """Make FoundationPose's tracker continue from this pose (object in the given camera's optical frame).
        fixed_camera=False (the wrist camera): only the tracker gets it. self.pose_cam is always in the FIXED camera's
        frame, and the tracking loop may publish it at any moment from its own thread (seen: one pose 702 mm off
        while the close look ran, because pose_cam briefly held the wrist-frame pose)."""
        import torch
        tf = self.est.get_tf_to_centered_mesh()
        self.est.pose_last = torch.as_tensor(pose_in_cam, dtype=torch.float, device="cuda") @ torch.linalg.inv(tf)
        if fixed_camera:
            self.pose_cam = pose_in_cam

    def _refine(self, req, resp):
        """Close look with the wrist camera: refine the object pose from a wrist frame taken after this request."""
        t_req = self.get_clock().now().nanoseconds + int(self.p("refine_settle_s") * 1e9)
        if req.initial.header.frame_id:
            q, t = req.initial.pose.orientation, req.initial.pose.position
            T_init = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])
        else:
            T_bc = self._T_base_cam(self.cam_frame) if self.cam_frame and self.pose_cam is not None else None
            if T_bc is None:
                resp.success, resp.message = False, "no initial pose (none given, none tracked)"
                return resp
            T_init = T_bc @ self.pose_cam
        end = time.time() + (req.timeout_s or 2.0) + self.p("refine_settle_s")
        frame = None
        while time.time() < end and frame is None:
            for d in reversed(self.w_depths):
                st = Time.from_msg(d.header.stamp).nanoseconds
                c = [r for r in self.w_rgbs if abs(Time.from_msg(r.header.stamp).nanoseconds - st) < 2e7]
                if st > t_req and c and self.w_info is not None:
                    frame = (img_to_np(c[-1]), img_to_np(d).astype(np.float32), d.header)
                    break
            if frame is None:
                time.sleep(0.03)
        if frame is None:
            resp.success = False
            newest = Time.from_msg(self.w_depths[-1].header.stamp).nanoseconds if self.w_depths else 0
            resp.message = (f"no fresh wrist depth frame (buffered: {len(self.w_depths)} depth, {len(self.w_rgbs)} "
                            f"color, info {'yes' if self.w_info else 'no'}; newest depth {(newest - t_req) / 1e9:+.2f} s "
                            f"vs the request)" + (" - wrist profile 'rgb': color only, no depth"
                                                  if self.wrist.get("profile") == "rgb" else ""))
            return resp
        rgb, depth, header = frame
        T_bw = self._T_base_cam(header.frame_id, header.stamp)
        if T_bw is None:
            resp.success, resp.message = False, f"no TF for {header.frame_id} at the image time"
            return resp
        K = np.array(self.w_info.k, float).reshape(3, 3)
        t0 = time.time()
        with self.est_lock:
            keep = (self.state, self.pose_cam)
            self._handover(np.linalg.inv(T_bw) @ T_init, fixed_camera=False)
            pose_w = self.est.track_one(rgb=rgb, depth=depth, K=K,
                                        iteration=int(req.iterations or self.p("refine_iterations")))
            T_ref = T_bw @ pose_w
            T_bc = self._T_base_cam(self.cam_frame) if self.cam_frame else None
            if T_bc is not None:                     # back to the fixed camera: tracking continues from the refined pose
                self._handover(np.linalg.inv(T_bc) @ T_ref)
            elif keep[1] is not None:                # no fixed-camera TF: continue tracking from where we were
                self._handover(keep[1])
        dR = T_init[:3, :3].T @ T_ref[:3, :3]
        resp.correction_m = float(np.linalg.norm(T_ref[:3, 3] - T_init[:3, 3]))
        resp.correction_deg = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        resp.pose.header.frame_id, resp.pose.header.stamp = self.base, header.stamp
        resp.pose.pose.position.x, resp.pose.pose.position.y, resp.pose.pose.position.z = map(float, T_ref[:3, 3])
        qr = matrix_to_quat(T_ref[:3, :3])
        o = resp.pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, qr)
        resp.success = True
        resp.message = (f"refined with the wrist camera in {(time.time() - t0) * 1000:.0f} ms: moved "
                        f"{resp.correction_m * 1000:.1f} mm, {resp.correction_deg:.1f} deg")
        self.reason = resp.message
        self.stats["last_refine"] = {"mm": round(resp.correction_m * 1000, 1), "deg": round(resp.correction_deg, 2)}
        self.get_logger().info(resp.message)
        return resp

    def _on_dets(self, msg):
        self.dets[Time.from_msg(msg.header.stamp).nanoseconds] = [
            (d.bbox.center.position.x, d.bbox.center.position.y) for d in msg.detections]
        while len(self.dets) > 30:
            self.dets.pop(next(iter(self.dets)))

    def _offset_to_detection(self, label, header):
        """xy distance [m] between the current pose and the detector's 3D estimate of detection `label` in the same
        image (None if that estimate isn't available)."""
        c = self.dets.get(Time.from_msg(header.stamp).nanoseconds)
        T_base_cam = self._T_base_cam(header.frame_id)
        if not c or label - 1 >= len(c) or T_base_cam is None:
            return None
        p = (T_base_cam @ self.pose_cam)[:3, 3]
        return float(np.hypot(p[0] - c[label - 1][0], p[1] - c[label - 1][1]))

    def _T_base_cam(self, cam_frame, stamp=None):
        try:
            if stamp is None:
                tf = self.tf.lookup_transform(self.base, cam_frame, Time()).transform
            else:                                   # a moving camera (wrist): its pose at the image's time
                tf = self.tf.lookup_transform(self.base, cam_frame, Time.from_msg(stamp),
                                              timeout=Duration(seconds=0.2)).transform
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"no TF {self.base} <- {cam_frame}: {e}", throttle_duration_sec=5.0)
            return None
        return make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                              [tf.translation.x, tf.translation.y, tf.translation.z])

    def _attached_tick(self):
        """ATTACHED: the object moves with the hand. No camera work at all (and none of the GPU load)."""
        try:
            ts = self.tf.lookup_transform(self.base, self.tcp, Time())
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"no TF {self.base} <- {self.tcp}: {e}", throttle_duration_sec=5.0)
            return
        tf = ts.transform
        T = make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                           [tf.translation.x, tf.translation.y, tf.translation.z]) @ self.T_tcp_obj
        T_base_cam = self._T_base_cam(self.cam_frame) if self.cam_frame else None
        if T_base_cam is not None:
            self.pose_cam = np.linalg.inv(T_base_cam) @ T
        # stamp = the time of the hand pose it came from (not "now"): the arm moves ~0.5 m/s, so 20 ms of
        # mislabeled time would already be 10 mm of apparent error for anyone comparing poses by time
        self._emit(T, ts.header.stamp)

    def _frame(self):
        """The newest mask (not processed yet) with the color + depth frames it was computed from, else None."""
        m = self.mask
        if m is None or self.info is None:
            return None
        stamp = Time.from_msg(m.header.stamp).nanoseconds
        if stamp == self.last_stamp or stamp < self.min_stamp:
            return None

        def match(buf):
            best = min(buf, key=lambda x: abs(Time.from_msg(x.header.stamp).nanoseconds - stamp), default=None)
            ok = best is not None and abs(Time.from_msg(best.header.stamp).nanoseconds - stamp) < 2e7
            return best if ok else None
        rgb, depth = match(self.rgbs), match(self.depths)
        if rgb is None or depth is None:
            return None
        self.last_stamp = stamp
        K = np.array(self.info.k, float).reshape(3, 3)
        return (img_to_np(rgb), img_to_np(depth).astype(np.float32), img_to_np(m), K, m.header)

    # --- the loop ----------------------------------------------------------------------------------------------------
    def _tick(self):
        if self.state == "ATTACHED":
            self._attached_tick()
            self._status()
            return
        f = self._frame()
        if f is None:
            return
        rgb, depth, labels, K, header = f
        self.cam_frame = header.frame_id
        target = self._target_label(labels, K)
        if target is not None and self.pose_cam is not None and self.state != "SEARCHING":
            off = self._offset_to_detection(target, header)
            if off is not None and off > self.p("max_offset_m"):
                target = None            # the detection under the pose is somewhere else in 3D: not where we are
        if self.state == "SEARCHING":
            if target:
                self._register(rgb, depth, labels == target, K, "first sight")
        elif target is None:                                    # nothing detected where the pose is
            if (labels > 0).any():                              # ... but the detector sees it somewhere else
                self.disagree += 1                              # (moved, or tracking slid off it)
                if self.state != "MISMATCH":
                    self.was = self.state
                # not TRACKING any more: until it re-registers, this pose is likely stale (e.g. the can was moved)
                # and a consumer waiting for TRACKING must not grasp at it
                self.state, self.reason = "MISMATCH", f"detector sees it elsewhere ({self.disagree} frame(s))"
                if self.disagree >= self.p("reregister_after"):
                    self._register(rgb, depth, labels == self._best_label(labels), K,
                                   "pose and detector disagree" if self.was == "TRACKING" else "reappeared elsewhere")
            else:
                self.state, self.reason = "HOLD", "target hidden: keeping the last pose"
        else:
            self.disagree = 0
            if self.state == "HOLD":
                self.reason = "target visible again at the held pose"
            self._track(rgb, depth, K)
        if self.pose_cam is not None:
            self._publish(header)
        self._status()

    def _best_label(self, labels):
        return 1 if (labels == 1).any() else int(labels.max())   # detector order = confidence order

    def _target_label(self, labels, K):
        """Which detection is OUR object: the one whose (dilated) mask contains the projected pose center.
        Before the first pose: the most confident detection."""
        if self.pose_cam is None:
            return self._best_label(labels) if (labels > 0).any() else None
        c = self.pose_cam[:3, 3]
        if c[2] <= 0:
            return None
        u, v = (K @ c)[:2] / c[2]
        r = int(self.p("mask_dilate_px"))
        h, w = labels.shape
        win = labels[max(0, int(v) - r):min(h, int(v) + r + 1), max(0, int(u) - r):min(w, int(u) + r + 1)]
        vals = win[win > 0]
        return int(np.bincount(vals).argmax()) if vals.size else None

    def _register(self, rgb, depth, mask, K, why):
        t, obj = time.time(), self.obj
        with self.est_lock:
            if self.obj != obj:                  # switched objects while waiting for the lock: this frame is stale
                return
            self.pose_cam = self.est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask.astype(np.uint8) * 255,
                                              iteration=int(FP_CFG["iterations"]["register"]))
        ms = (time.time() - t) * 1000
        self.stats["registrations"] += 1
        self.stats["last_register_ms"] = round(ms)
        self.state, self.reason, self.disagree = "TRACKING", f"registered ({why})", 0
        self.get_logger().info(f"registered '{self.obj}' in {ms:.0f} ms ({why})")

    def _track(self, rgb, depth, K):
        t, obj = time.time(), self.obj
        with self.est_lock:
            if self.obj != obj:
                return
            self.pose_cam = self.est.track_one(rgb=rgb, depth=depth, K=K, iteration=int(FP_CFG["iterations"]["track"]))
        self.stats["tracked_frames"] += 1
        self.stats["last_track_ms"] = round((time.time() - t) * 1000)
        self.state = "TRACKING"

    # --- outputs ------------------------------------------------------------------------------------------------------
    def _publish(self, header):
        T_base_cam = self._T_base_cam(header.frame_id)
        if T_base_cam is not None:
            self._emit(T_base_cam @ self.pose_cam, header.stamp)

    def _status(self):
        """State + counters, AND the pose this state belongs to: consumers that act on the pose (pick & place) read
        both from one message. Pairing /perception/object_pose with a separately received status can match a stale
        pose with the previous tick's "TRACKING"."""
        d = {"state": self.state, "reason": self.reason, "object": self.obj, **self.stats}
        if self.last_emit is not None:
            d["T_base_obj"], d["stamp"] = self.last_emit
        self.pub_status.publish(String(data=json.dumps(d)))
        self.last_emit = None

    def _emit(self, T, stamp):
        """Publish the pose T (base frame) + its RViz markers."""
        self.last_emit = (np.asarray(T).round(6).tolist(), stamp.sec + stamp.nanosec * 1e-9)
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, self.base
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, T[:3, 3])
        q = matrix_to_quat(T[:3, :3])
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
        self.pub_pose.publish(ps)

        ms = MarkerArray()
        mesh = Marker(header=ps.header, ns="object", id=0, type=Marker.MESH_RESOURCE, pose=ps.pose)
        mesh.mesh_resource = f"file://{self.mesh_path}"
        mesh.mesh_use_embedded_materials = True
        mesh.scale.x = mesh.scale.y = mesh.scale.z = 1.0
        # white = tracked from the images, orange = held while hidden, violet = carried (pose from the gripper)
        mesh.color.r, mesh.color.g, mesh.color.b, mesh.color.a = {
            "HOLD": (1.0, 0.6, 0.2, 0.6), "ATTACHED": (0.75, 0.5, 1.0, 0.8)}.get(self.state, (1.0, 1.0, 1.0, 1.0))
        ms.markers.append(mesh)
        for k, col in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1))):     # object axes x/y/z = red/green/blue
            ax = Marker(header=ps.header, ns="axes", id=k, type=Marker.ARROW)
            a = T[:3, 3]
            b = a + T[:3, k] * 0.08
            ax.points = [Point(x=float(a[0]), y=float(a[1]), z=float(a[2])), Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))]
            ax.scale.x, ax.scale.y, ax.scale.z = 0.004, 0.008, 0.01
            ax.color.r, ax.color.g, ax.color.b, ax.color.a = (*map(float, col), 1.0)
            ms.markers.append(ax)
        txt = Marker(header=ps.header, ns="state", id=0, type=Marker.TEXT_VIEW_FACING)
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(T[0, 3]), float(T[1, 3]), float(T[2, 3]) + 0.12
        txt.scale.z = 0.025
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
        txt.text = f"6D pose: {self.state}" + (f"  track {self.stats['last_track_ms']} ms" if self.stats["last_track_ms"] else "")
        ms.markers.append(txt)
        self.pub_mk.publish(ms)


def main():
    rclpy.init()
    node = PoseEstimatorNode()
    ex = MultiThreadedExecutor(num_threads=3)       # the close-look service waits for frames in its own thread
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
