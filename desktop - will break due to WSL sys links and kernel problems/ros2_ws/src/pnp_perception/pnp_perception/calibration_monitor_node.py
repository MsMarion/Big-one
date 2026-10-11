"""Is the fixed camera still where its calibration says? (docs/learning/16_calibration.md)

The AprilTag(s) on the table (config/calibration_targets.yaml) never move. If one APPEARS to move, the camera did
(a bumped stand, a sagging mount, thermal drift), and every 3-D measurement from it is now off by about as much.
Every period_s: detect the tag in the fixed camera, project its REFERENCE corners (its base-frame pose recorded right
after the last calibration: config/calibration/<set>_d435i.yaml apriltag_reference) through the CURRENT camera TF,
and measure how many pixels they are from where the tag is seen now (median over the last `window` checks).
Pixels are the camera's own unit: corner detection is good to ~0.1 px, and 1 px at 1 m ~ 1.6 mm sideways or
0.09 deg of camera rotation.

  /calibration/status  std_msgs/String JSON, latched:
                       {state: OK | WARN | DRIFT | NO_TAG | NO_REFERENCE, px, mm, tags, reference}
OK -> WARN at warn_px, DRIFT at drift_px. pick_place_node reads it (param calibration_policy: warn | stop).
A new calibration (/calibration/state changes) re-reads the reference. tools/calibrate.py reference re-records it.

Re-anchor (param reanchor: off | auto; the scene's perception.calibration_monitor.reanchor sets the default): the
tags never move, so the camera pose that maps their REFERENCE corners onto what is seen now (one solvePnP over every
visible tag's corners) is where the camera is NOW. If that pose is steady over the window (reanchor_steady_mm) and
differs from the calibration by more than reanchor_min_mm but at most reanchor_max_mm / reanchor_max_deg (a bumped
stand, thermal creep), it becomes the calibration: written like tools/calibrate.py writes one (history kept, the tag
reference kept), /calibration/reload, an event in the trial log. Beyond the bound it is NOT corrected: DRIFT stays
and a real calibration is needed (tools/calibrate.py locate / fixed). Needs >= reanchor_min_tags tags in view.
"""

import collections
import datetime
import json
import time

import cv2

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from pnp_core import trials
from pnp_core.calibration import detect_tags, drift_state, image_to_gray, pose_error, project, tag_corners
from pnp_core.catalog import (calibration_dir, calibration_targets_path, load_fixed_camera, load_robot, load_scene,
                              load_yaml)
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix


class CalibrationMonitor(Node):
    def __init__(self):
        super().__init__("pnp_calibration_monitor")
        self.declare_parameter("period_s", 1.0)
        self.declare_parameter("window", 5)
        self.declare_parameter("warn_px", 1.5)
        self.declare_parameter("drift_px", 3.0)
        self.declare_parameter("no_tag_s", 10.0)       # no tag seen this long -> NO_TAG (covered? lights off?)
        mon = load_scene().get("perception", {}).get("calibration_monitor", {})
        self.declare_parameter("reanchor", str(mon.get("reanchor", "off")))     # off | auto (see the top of this file)
        self.declare_parameter("reanchor_min_mm", 1.0)       # below: noise, leave the calibration alone
        self.declare_parameter("reanchor_max_mm", 5.0)       # above (or reanchor_max_deg): a real move -> recalibrate
        self.declare_parameter("reanchor_max_deg", 0.5)
        self.declare_parameter("reanchor_steady_mm", 0.5)    # the window's PnP poses agree this well: not mid-bump
        self.declare_parameter("reanchor_min_tags", 3)
        self.anchor_hist = collections.deque(maxlen=int(self.get_parameter("window").value))
        self.reload_cli = self.create_client(Trigger, "/calibration/reload")
        self.refused_at = 0.0
        self.tags_cfg = load_yaml(calibration_targets_path())["apriltags"]
        self.cam = load_fixed_camera()
        self.base = load_robot()["frames"]["base"]
        topics = load_yaml("config/interfaces.yaml")["topics"]
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.img, self.info, self.last_stamp = None, None, None
        self.create_subscription(Image, topics["color_image"]["name"], lambda m: setattr(self, "img", m),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, topics["color_info"]["name"], lambda m: setattr(self, "info", m),
                                 qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(String, "/calibration/status", latched)
        self.hist = collections.deque(maxlen=int(self.get_parameter("window").value))
        self.state, self.last_seen = None, time.time()
        self.ref, self.ref_date, self.cal_set = {}, None, None
        # the ACTIVE calibration set comes from camera_tf_node (a new calibration or set -> re-read the reference)
        self.create_subscription(String, "/calibration/state", self._on_cal_state, latched)
        self.create_timer(float(self.get_parameter("period_s").value), self._check)

    def _on_cal_state(self, msg):
        self.cal_set = json.loads(msg.data).get("set")
        self._load_reference()

    def _load_reference(self):
        p = calibration_dir() / f"{self.cal_set}_{self.cam['name']}.yaml"
        c = yaml.safe_load(p.read_text()) if p.exists() else {}
        ref = c.get("apriltag_reference") or {}
        self.ref = {int(k): make_transform(quat_to_matrix(v["quat_xyzw"]), v["xyz"]) for k, v in ref.items()}
        self.ref_date = c.get("apriltag_reference_date")
        self.hist.clear()                    # a new calibration: old offsets mean nothing now
        self.anchor_hist.clear()
        self.get_logger().info(f"tag reference: {sorted(self.ref) or 'none'} ({p.name}, {self.ref_date})")

    def _publish(self, state, **kw):
        msg = {"state": state, "reference": self.ref_date, **kw}
        self.pub.publish(String(data=json.dumps(msg)))
        if state != self.state:
            # (two separate calls: rclpy raises if ONE logging call site is used with different severities)
            text = f"calibration status {self.state} -> {state} {kw}"
            if state in ("WARN", "DRIFT"):
                self.get_logger().warn(text)
            else:
                self.get_logger().info(text)
            # every state change -> the events log (pnp_core/trials.py), with how far off it was
            trials.event("calibration_monitor", "calibration_state", {"from": self.state, "to": state, **kw},
                         "warn" if state in ("WARN", "DRIFT", "NO_TAG") else "info")
            self.state = state

    def _check(self):
        if not self.ref:
            self._publish("NO_REFERENCE", set=self.cal_set,
                          note="no calibration with a tag reference: tools/calibrate.py fixed (or: calibrate.py reference)")
            return
        m, info = self.img, self.info
        if m is None or info is None or m.header.stamp == self.last_stamp:
            return
        self.last_stamp = m.header.stamp
        try:
            T_base_cam = self._tf(m.header.frame_id)
        except Exception:  # noqa: BLE001
            return
        K, D = np.array(info.k).reshape(3, 3), np.array(info.d, float) if len(info.d) else np.zeros(5)
        seen = detect_tags(image_to_gray(m), self.tags_cfg["dictionary"], self.tags_cfg["size_m"], K, D,
                           with_corners=True)
        corners_t = tag_corners(self.tags_cfg["size_m"])
        per_tag = {}
        for tid, T_ref in self.ref.items():
            if tid in seen:
                T_cam_tag = np.linalg.inv(T_base_cam) @ T_ref
                px = float(np.mean(np.linalg.norm(project(T_cam_tag, corners_t, K, D) - seen[tid][1], axis=1)))
                per_tag[tid] = (px, px * T_cam_tag[2, 3] / K[0, 0] * 1000)       # pixels, mm at the tag
        if not per_tag:
            if time.time() - self.last_seen > self.get_parameter("no_tag_s").value:
                self._publish("NO_TAG", note="reference tag not visible (covered? out of view?)")
            return
        self.last_seen = time.time()
        self.hist.append(max(per_tag.values()))                       # the worst tag: (px, mm)
        px, mm = (float(v) for v in np.median(np.array(self.hist), axis=0))
        state = drift_state(px, self.get_parameter("warn_px").value, self.get_parameter("drift_px").value)
        self._publish(state, set=self.cal_set, px=round(px, 2), mm=round(mm, 2),
                      tags={str(k): round(v[0], 2) for k, v in per_tag.items()})
        if self.get_parameter("reanchor").value == "auto":
            self._reanchor(seen, corners_t, K, D, T_base_cam, m.header.frame_id)

    def _reanchor(self, seen, corners_t, K, D, T_base_cam, optical):
        """Where is the camera NOW, from every visible reference tag's corners; a small, steady difference from the
        calibration -> it becomes the calibration (see the top of this file)."""
        ids = [t for t in self.ref if t in seen]
        if len(ids) < int(self.get_parameter("reanchor_min_tags").value):
            self.anchor_hist.clear()
            return
        P3 = np.vstack([(corners_t @ self.ref[t][:3, :3].T + self.ref[t][:3, 3]) for t in ids])   # base frame
        P2 = np.vstack([seen[t][1] for t in ids]).astype(np.float64)
        T_cam_base0 = np.linalg.inv(T_base_cam)
        rvec0, _ = cv2.Rodrigues(T_cam_base0[:3, :3])
        ok, rvec, tvec = cv2.solvePnP(P3, P2, K, D, rvec0, T_cam_base0[:3, 3].reshape(3, 1), useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return
        T_cam_base = make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel())
        self.anchor_hist.append(np.linalg.inv(T_cam_base))
        if len(self.anchor_hist) < self.anchor_hist.maxlen:
            return
        Ts = list(self.anchor_hist)
        if max(pose_error(Ts[-1], T)[0] for T in Ts) > self.get_parameter("reanchor_steady_mm").value:
            return                                                       # still moving: wait
        T_now = Ts[-1]
        mm, deg = pose_error(T_base_cam, T_now)
        if mm < self.get_parameter("reanchor_min_mm").value and deg < 0.05:
            return
        if mm > self.get_parameter("reanchor_max_mm").value or deg > self.get_parameter("reanchor_max_deg").value:
            if time.time() - self.refused_at > 30.0:
                self.refused_at = time.time()
                self.get_logger().warn(f"re-anchor refused: the camera moved {mm:.1f} mm / {deg:.2f} deg (more than "
                                       f"the bound): recalibrate (tools/calibrate.py locate / fixed)")
                trials.event("calibration_monitor", "reanchor_refused",
                             {"mm": round(mm, 2), "deg": round(deg, 3), "tags": ids}, "warn")
            return
        self._write_anchor(T_now, mm, deg, ids, optical)

    def _write_anchor(self, T_base_opt, mm, deg, ids, optical):
        path = calibration_dir() / f"{self.cal_set}_{self.cam['name']}.yaml"
        if not path.exists():
            return
        c = yaml.safe_load(path.read_text())
        if c.get("parent") != self.base or c.get("optical_frame") != optical:
            self.get_logger().warn(f"re-anchor: {path.name} is not {self.base} -> {optical}: not written")
            return
        old = {k: c[k] for k in ("date", "T_parent_optical", "method", "reprojection_px") if k in c}
        c["history"] = ([old] + list(c.get("history") or []))[:10]
        c["T_parent_optical"] = {"xyz": [round(float(v), 6) for v in T_base_opt[:3, 3]],
                                 "quat_xyzw": [round(float(v), 7) for v in matrix_to_quat(T_base_opt[:3, :3])]}
        c["date"] = datetime.datetime.now().isoformat(timespec="seconds")
        c["method"] = f"reanchor (AprilTags {sorted(ids)}, moved {mm:.2f} mm / {deg:.3f} deg)"
        head = (f"# Calibration of the {c.get('camera', self.cam['name'])} (re-anchored by calibration_monitor_node). "
                f"camera_tf_node publishes\n# T_parent_optical as {c['parent']} -> {c['optical_frame']}. history: "
                f"earlier results, newest first.\n")
        path.write_text(head + yaml.safe_dump(c, sort_keys=False, default_flow_style=None, width=120))
        self.get_logger().warn(f"re-anchored: the camera moved {mm:.2f} mm / {deg:.3f} deg; {path.name} updated")
        trials.event("calibration_monitor", "reanchor", {"mm": round(mm, 2), "deg": round(deg, 3), "tags": sorted(ids),
                                                         "file": path.name})
        self.anchor_hist.clear()
        if self.reload_cli.service_is_ready():
            self.reload_cli.call_async(Trigger.Request())       # camera_tf_node republishes -> the reference reloads

    def _tf(self, frame):
        t = self.tf.lookup_transform(self.base, frame, Time()).transform
        return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                              [t.translation.x, t.translation.y, t.translation.z])


def main():
    rclpy.init()
    n = CalibrationMonitor()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
