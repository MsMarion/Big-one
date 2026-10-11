"""Perception node: camera images -> 3D object positions in the robot base frame.

    ros2 run pnp_perception detector_node --ros-args -p model:=/repo/models/soup_can_seg_v1.pt

Pipeline per processed frame (continuous, `rate_hz`):
  color + aligned depth (time-synced)  --YOLO-seg-->  instance masks
  mask + depth + camera_info  --pnp_core.estimate_object-->  can axis (x, y), top z, fit quality
  accepted estimates  -->  /perception/detections   (vision_msgs/Detection3DArray, frame fr3_link0)
                      -->  /perception/markers      (RViz: cylinders + labels)
                      -->  /perception/debug_image  (masks drawn: green = accepted, red = rejected + reason)
On demand:  /perception/detect  (pnp_interfaces/DetectObjects) waits for a frame captured AFTER the request,
so the pick logic never acts on a stale estimate.

Several objects (scenes with extra_objects, the workcell page): every class the model knows that is an object in
config/objects.yaml (its yolo_class) is detected, each with ITS size and shape. The FOCUS object (std_msgs/String
/perception/focus, latched: the object being picked; default the `object` parameter) is the only one in
/perception/mask, so the 6-D pose estimator registers on it and not on its neighbour.

All geometry lives in pnp_core (unit-tested without ROS); this file is only the ROS wrapper.
"""

import copy
import threading
import time

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from pnp_interfaces.srv import DetectObjects
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

from pnp_core.focus import nearest, parse_focus  # noqa: E402
from pnp_core.catalog import load_camera, load_object, load_robot, load_scene, load_yaml
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix, rot_z
from pnp_core.perception_math import Intrinsics, Quality, depth_to_meters, estimate_object
from pnp_core.vision import CatalogSize, DepthBlobs, Frame


class DetectorNode(Node):
    def __init__(self):
        super().__init__("pnp_detector")
        self.declare_parameter("model", "/repo/models/soup_can_seg_v1.pt")
        self.declare_parameter("object", "tomato_soup_can")      # config/objects.yaml entry
        self.declare_parameter("conf", 0.5)
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("table_z", 0.0)
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        from ultralytics import YOLO   # imported here: heavy, and only needed once the node starts
        self.model = YOLO(p("model"))
        self.device, self.conf = p("device"), p("conf")
        self.obj_name = p("object")
        self.obj = load_object(self.obj_name)
        self.label = self.obj["yolo_class"]
        # every object the model can see: yolo_class -> (objects.yaml name, its config)
        names = set(self.model.names.values())
        self.by_label = {o["yolo_class"]: (n, o) for n, o in load_yaml("config/objects.yaml").items()
                         if o.get("yolo_class") in names}
        self.by_label[self.label] = (self.obj_name, self.obj)
        self.focus_label = self.label
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, "/perception/focus", self._on_focus, latched)
        self.table_z = p("table_z")
        # the cell may tune the detector (perception.detector in the scene): a desk with 1 cm bricks needs a tighter
        # height check than a table with 10 cm cans, and nothing may be "found" on the robot's own base
        scene = load_scene()
        t = scene["table"]
        self.table_z = t["center"][2] + t["size"][2] / 2          # the scene's table top (the SO-101 desk: -2.4 mm)
        tune = scene.get("perception", {}).get("detector", {})
        self.conf = float(tune.get("conf", self.conf))
        self.quality = Quality(**{k: tune[k] for k in ("min_points", "max_fit_residual_m", "height_tolerance_m")
                                  if k in tune})
        self.exclude_r = float(tune.get("exclude_radius_m", 0.0))     # around the robot base (its own parts)
        if tune:
            self.get_logger().info(f"scene tuning: {tune}")
        # objects the model has no class for (perception.detector.size_classes, e.g. lego_2x3): found by depth
        # geometry instead -- blobs above the desk (pnp_core.vision.DepthBlobs) whose footprint + height match that
        # catalog entry best among ALL bricks (CatalogSize, 6 mm). A model detection on the same spot is replaced
        # (the 2x3 is the 2x4's red: the model calls it a 2x4)
        catalog = load_yaml("config/objects.yaml")
        self.size_classes = {catalog[n]["yolo_class"]: (n, catalog[n]) for n in tune.get("size_classes", [])
                             if n in catalog}
        if self.size_classes:
            # the arm's reach only, every depth pixel (stride 2 found nothing: too few points per 2 mm cell). ~0.7 s
            # a pass: its own 1 Hz timer on its own thread (the detection loop uses the latest result, not waiting)
            r = float(scene.get("ui", {}).get("reach_m", [0.0, 0.3])[1]) + 0.03
            self.blobs = DepthBlobs((0.0, r, -r, r), self.table_z, stride=1)
            self.blob_cache, self.blob_input = [], None
            self.sizer = CatalogSize(catalog)
            self.by_label.update(self.size_classes)
        cam = load_camera()
        topics = load_yaml("config/interfaces.yaml")["topics"]
        self.base = load_robot()["frames"]["base"]
        self.optical = cam["frames"]["color_optical"]
        self.bridge = CvBridge()
        self.tf = Buffer()
        TransformListener(self.tf, self)

        self.lock = threading.Lock()
        self.latest = None            # (color_msg, depth_msg) waiting to be processed
        self.info = None
        self.last_result = None       # (stamp, Detection3DArray)
        self.result_event = threading.Condition()

        sensor = qos_profile_sensor_data
        self.create_subscription(CameraInfo, topics["color_info"]["name"], self._info, sensor)
        color = message_filters.Subscriber(self, Image, topics["color_image"]["name"], qos_profile=sensor)
        depth = message_filters.Subscriber(self, Image, topics["aligned_depth"]["name"], qos_profile=sensor)
        # The link downsamples color and depth independently, so a pair can be one frame (33 ms) apart.
        sync = message_filters.ApproximateTimeSynchronizer([color, depth], queue_size=10, slop=0.05)
        sync.registerCallback(self._pair)

        self.pub_det = self.create_publisher(Detection3DArray, topics["detections"]["name"], 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/perception/markers", 10)
        self.pub_dbg = self.create_publisher(Image, "/perception/debug_image", sensor)
        # instance masks of the ACCEPTED detections as one label image (mono8: 0 = background, k = detection k-1),
        # same stamp as the color frame: input of the 6-D pose estimator (pose_estimator_node, FoundationPose)
        self.pub_mask = self.create_publisher(Image, "/perception/mask", sensor)
        self.create_timer(1.0 / p("rate_hz"), self._process, callback_group=MutuallyExclusiveCallbackGroup())
        if self.size_classes:
            self.create_timer(1.0, self._update_size_blobs, callback_group=MutuallyExclusiveCallbackGroup())
        self.create_service(DetectObjects, "/perception/detect", self._detect, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(f"detector ready: model={p('model')} classes={sorted(self.by_label)} focus={self.label} "
                               f"conf>={self.conf}")

    # --- inputs -------------------------------------------------------------------------------------
    def _on_focus(self, msg):
        name, self.focus_hint = parse_focus(msg.data)          # pnp_core/focus.py: + "near here" (which instance)
        o = load_yaml("config/objects.yaml").get(name)
        if o and o.get("yolo_class") in self.by_label:
            self.focus_label = o["yolo_class"]
            near = f" near ({self.focus_hint[0]:.3f}, {self.focus_hint[1]:.3f})" if self.focus_hint else ""
            self.get_logger().info(f"focus: {name} ({self.focus_label}){near}")

    def _info(self, msg):
        self.info = msg

    def _pair(self, color, depth):
        with self.lock:
            self.latest = (color, depth)

    # --- processing ---------------------------------------------------------------------------------
    def _process(self):
        with self.lock:
            pair, self.latest = self.latest, None
        if pair is None or self.info is None:
            return
        color_msg, depth_msg = pair
        stamp = Time.from_msg(color_msg.header.stamp)
        try:
            tf = self.tf.lookup_transform(self.base, self.optical, Time())   # static mount (sim) / calibration (real)
        except Exception as e:   # noqa: BLE001
            self.get_logger().warn(f"no TF {self.base} <- {self.optical}: {e}", throttle_duration_sec=5.0)
            return
        t, q = tf.transform.translation, tf.transform.rotation
        T_base_cam = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])
        K = Intrinsics.from_k(self.info.k)
        img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
        depth = depth_to_meters(self.bridge.imgmsg_to_cv2(depth_msg, "passthrough"), depth_msg.encoding)

        res = self.model.predict(img, conf=self.conf, device=self.device, retina_masks=True, verbose=False)[0]
        self.blob_input = (depth, T_base_cam)
        by_size = self.blob_cache if self.size_classes else []        # [(label, mask, xy)], from _update_size_blobs
        out = Detection3DArray()
        out.header.stamp, out.header.frame_id = color_msg.header.stamp, self.base
        debug = img.copy()
        labels = np.zeros(img.shape[:2], np.uint8)
        focus = []                                       # (detection index, mask, xy) of the focus class
        if res.masks is not None:
            for k, (m, box, score, cls) in enumerate(zip(res.masks.data.cpu().numpy(), res.boxes.xyxy.cpu().numpy(),
                                                          res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy())):
                label = self.model.names[int(cls)]
                if label not in self.by_label:
                    continue
                obj = self.by_label[label][1]
                mask = m > 0.5
                est, why = estimate_object(mask, depth, K, T_base_cam, obj, self.table_z, self.quality)
                if est is not None and any(np.hypot(est.center_xy[0] - b[2][0], est.center_xy[1] - b[2][1]) < 0.015
                                           for b in by_size):
                    continue                                  # a size-class object: reported from its blob below
                if est is not None and np.hypot(*est.center_xy[:2]) < self.exclude_r:
                    est, why = None, f"on the robot ({np.hypot(*est.center_xy[:2]):.2f} m from its base)"
                ok = est is not None
                color = (0, 220, 0) if ok else (0, 0, 255)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug, contours, -1, color, 2)
                x0, y0 = int(box[0]), max(int(box[1]) - 6, 12)
                text = (f"{label} {score:.2f} ({est.center_xy[0]:.3f},{est.center_xy[1]:.3f})" if ok
                        else f"{label} {score:.2f} REJECT: {why}")
                cv2.putText(debug, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                if ok:
                    out.detections.append(self._detection(out.header, est, float(score), len(out.detections), label, obj))
                    if label == self.focus_label:            # the pose estimator's input: the focus object only
                        focus.append((len(out.detections), mask, est.center_xy[:2]))
        for label, mask, _ in by_size:
            obj = self.by_label[label][1]
            est, why = estimate_object(mask, depth, K, T_base_cam, obj, self.table_z, self.quality)
            if est is None or np.hypot(*est.center_xy[:2]) < self.exclude_r:
                continue
            out.detections.append(self._detection(out.header, est, 0.9, len(out.detections), label, obj))
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug, contours, -1, (220, 160, 0), 2)
            if label == self.focus_label:
                focus.append((len(out.detections), mask, est.center_xy[:2]))
        # with a hint (the object clicked / being picked) only THAT instance: with two of a class the pose estimator
        # would otherwise register on whichever mask came first
        for idx, mask, _ in nearest(focus, getattr(self, "focus_hint", None), key=lambda f: f[2]):
            labels[mask & (labels == 0)] = idx
        self.pub_det.publish(out)
        self.pub_mk.publish(self._markers(out))
        dbg = self.bridge.cv2_to_imgmsg(debug, "bgr8")
        dbg.header = color_msg.header
        self.pub_dbg.publish(dbg)
        lab = self.bridge.cv2_to_imgmsg(labels, "mono8")
        lab.header = color_msg.header
        self.pub_mask.publish(lab)
        with self.result_event:
            self.last_result = (stamp, out)
            self.result_event.notify_all()

    def _update_size_blobs(self):
        """Blobs on the desk that match a size class -> self.blob_cache [(label, pixel mask, center xy)]."""
        if self.blob_input is None or self.info is None:
            return
        depth, T_base_cam = self.blob_input
        try:
            frame = Frame(None, depth, np.asarray(self.info.k, float).reshape(3, 3), T_base_cam)
            out = []
            for p in self.blobs.propose(frame):
                ident = self.sizer.identify(frame, p)
                if ident is None or ident.extra.get("rest") not in (None, "upright"):
                    continue
                label = self.by_label_of(ident.name)
                if label in self.size_classes:
                    out.append((label, p.mask, p.center_xy))
            self.blob_cache = out
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"size classes: {e}", throttle_duration_sec=10.0)

    def by_label_of(self, name):
        return next((lab for lab, (n, _) in self.by_label.items() if n == name), None)

    def _detection(self, header, est, score, k, label, obj):
        h = est.height_m or obj["dims_m"][2]                         # the face it rests on decides its height
        d = Detection3D()
        d.header, d.id = header, f"{label}_{k}"
        pose = Pose()
        pose.position.x, pose.position.y = float(est.center_xy[0]), float(est.center_xy[1])
        pose.position.z = float(est.z_top - h / 2)                      # object center
        qx, qy, qz, qw = matrix_to_quat(rot_z(est.yaw))
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
        d.bbox.center = pose
        if est.footprint_m and abs(h - obj["dims_m"][2]) > 1e-6:      # on another face: its footprint there
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = float(est.footprint_m[0]), float(est.footprint_m[1]), float(h)
        else:
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = map(float, obj["dims_m"])
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id, hyp.hypothesis.score = label, score
        hyp.pose.pose = copy.deepcopy(pose)
        d.results.append(hyp)
        return d

    def _markers(self, arr):
        ms = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ms.markers.append(clear)
        for k, d in enumerate(arr.detections):
            m = Marker()
            label = d.results[0].hypothesis.class_id
            box = self.by_label.get(label, (None, self.obj))[1]["shape"] == "box"   # a box at its yaw, else a cylinder
            m.header, m.ns, m.id, m.type = arr.header, "detections", k, Marker.CUBE if box else Marker.CYLINDER
            m.pose = d.bbox.center
            m.scale.x, m.scale.y = d.bbox.size.x, (d.bbox.size.y if box else d.bbox.size.x)
            m.scale.z = d.bbox.size.z
            m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 0.9, 0.2, 0.5) if label == self.focus_label else \
                (0.2, 0.5, 1.0, 0.45)                    # green: the focus object, blue: the others
            t = Marker()
            t.header, t.ns, t.id, t.type = arr.header, "labels", k, Marker.TEXT_VIEW_FACING
            # COPY the pose: `t.pose = d.bbox.center` would alias the detection's own Pose object, and raising
            # the label would silently move the detection up by one can height (found by eval_perception.py).
            t.pose = copy.deepcopy(d.bbox.center)
            t.pose.position.z += d.bbox.size.z
            t.scale.z = 0.03
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = f"{d.results[0].hypothesis.class_id} {d.results[0].hypothesis.score:.2f}"
            ms.markers += [m, t]
        return ms

    # --- on-demand service --------------------------------------------------------------------------
    def _detect(self, req, resp):
        asked = self.get_clock().now()
        deadline = time.time() + (req.timeout_s or 2.0)
        with self.result_event:
            while not (self.last_result and self.last_result[0] > asked):
                remaining = deadline - time.time()
                if remaining <= 0:
                    resp.success, resp.message = False, "no fresh camera frame in time (is the camera/sim running?)"
                    return resp
                self.result_event.wait(timeout=remaining)
            arr = self.last_result[1]
        if req.object_class:
            arr.detections = [d for d in arr.detections if d.results[0].hypothesis.class_id == req.object_class]
        resp.success, resp.detections = True, arr
        resp.message = f"{len(arr.detections)} detection(s)"
        return resp


def main():
    rclpy.init()
    node = DetectorNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
