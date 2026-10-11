"""Calibration system test: do the robot and both cameras agree about where things are? (docs/learning/16_calibration.md)

    python3 /repo/tools/calibration_check.py            ~30 s; the arm moves the wrist camera over the AprilTag
        --no-move                                       fixed-camera checks only (the arm stays where it is)

Run it at the start of a session, after touching a camera, or when picks start missing. Each check has a limit:
  tag drift      fixed camera: the AprilTag vs its reference from the last calibration   (has the camera moved?)
  table plane    fixed camera depth: the table top must come out at z = 0 and level      (pose + depth together)
  depth scale    measured table depth vs the depth the calibrated pose predicts          (depth sensor scale)
  cross-camera   the wrist camera finds the same tag: its 3-D tag center must lie on the fixed camera's ray
                 through the tag center                                                  (both calibrations + arm)
  wrist depth    the depth-scale test for the D405, at the tag view
  truth (sim)    both camera poses in TF vs the TRUE poses in Isaac
-> PASS / FAIL per line, exit code 0 = all passed, a JSON report in data/calibration/.
"""

import argparse
import datetime
import json
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402

from pnp_core.calibration import (detect_tags, eye_in_hand_poses, image_to_gray, mean_transform,  # noqa: E402
                                  pose_error, project, tag_corners)
from pnp_core.catalog import calibration_dir, calibration_targets_path, load_fixed_camera, load_robot, load_yaml, repo_root  # noqa: E402
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix  # noqa: E402
from pnp_core.object_grasp import top_down_R  # noqa: E402
from pnp_core.perception_math import depth_to_meters  # noqa: E402
from pnp_core.wrist import wrist_camera  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402

LIMITS = {"tag_px": 1.5, "table_mm": 3.0, "table_deg": 0.3, "depth_pct": 1.0, "cross_mm": 2.0,
          "truth_mm": 2.0, "truth_deg": 0.2}


class Cam:
    """Latest color / info / depth of one camera."""
    def __init__(self, node, color, info, depth):
        self.m = {"n": 0}
        node.create_subscription(Image, color, self._color, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, info, lambda x: self.m.__setitem__("info", x), qos_profile_sensor_data)
        node.create_subscription(Image, depth, lambda x: self.m.__setitem__("depth", x), qos_profile_sensor_data)

    def _color(self, x):
        self.m["color"] = x
        self.m["n"] += 1

    def fresh(self, skip=2, timeout=5.0):
        n0, end = self.m["n"], time.time() + timeout
        while (self.m["n"] < n0 + skip or "info" not in self.m or "depth" not in self.m) and time.time() < end:
            time.sleep(0.02)
        if self.m["n"] < n0 + skip:
            return None
        info = self.m["info"]
        return {"color": self.m["color"], "depth": self.m["depth"], "K": np.array(info.k).reshape(3, 3),
                "D": np.array(info.d, float) if len(info.d) else np.zeros(5)}


def depth_m(img) -> np.ndarray:
    a = np.frombuffer(img.data, np.uint16 if img.encoding == "16UC1" else np.float32).reshape(img.height, img.width)
    return depth_to_meters(a, img.encoding)


class Checker:
    def __init__(self, args):
        self.args = args
        self.node = Node("pnp_calibration_check")
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(self.node)
        threading.Thread(target=ex.spin, daemon=True).start()
        self.robot_cfg = load_robot()
        self.base = self.robot_cfg["frames"]["base"]
        self.R = RobotInterface(self.node, self.robot_cfg)
        t = load_yaml("config/interfaces.yaml")["topics"]
        self.fixed = Cam(self.node, t["color_image"]["name"], t["color_info"]["name"], t["aligned_depth"]["name"])
        self.wcfg = wrist_camera(self.robot_cfg)          # None: the robot has no wrist camera (fixed checks only)
        self.wrist = Cam(self.node, self.wcfg["topics"]["color_image"], self.wcfg["topics"]["color_info"],
                         self.wcfg["topics"]["aligned_depth"]) if self.wcfg else None
        self.tags = load_yaml(calibration_targets_path())["apriltags"]
        self.bench = Bench(name="pnp_calibration_check_sim")
        self.sim = self.bench.sim.wait_ready(timeout=3.0)
        self.rows, self.report = [], {}
        self.cal_state = {}          # camera_tf_node's /calibration/state (the active calibration set)
        self.node.create_subscription(String, "/calibration/state", lambda m: setattr(self, "cal_state", json.loads(m.data)),
                                      QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                                 reliability=ReliabilityPolicy.RELIABLE))

    def tf(self, parent, child, stamp=None):
        end = time.time() + 5.0
        while True:
            for t in ([Time.from_msg(stamp)] if stamp else []) + [Time()]:
                try:
                    x = self.R.tf.lookup_transform(parent, child, t).transform
                    return make_transform(quat_to_matrix([x.rotation.x, x.rotation.y, x.rotation.z, x.rotation.w]),
                                          [x.translation.x, x.translation.y, x.translation.z])
                except Exception:  # noqa: BLE001
                    pass
            if time.time() > end:
                raise RuntimeError(f"no TF {parent} -> {child}")
            time.sleep(0.1)

    def row(self, name, value, limit, unit, ok=None, note=""):
        ok = (value <= limit) if ok is None else ok
        self.rows.append((name, value, limit, unit, "PASS" if ok else "FAIL", note))
        self.report[name] = {"value": round(float(value), 3), "limit": limit, "unit": unit, "pass": bool(ok), "note": note}
        print(f"  {name:<22} {value:8.2f} {unit:<4} (limit {limit:g})  {'PASS' if ok else 'FAIL'}  {note}", flush=True)

    def skip(self, name, why):
        self.rows.append((name, None, None, "", "SKIP", why))
        self.report[name] = {"skipped": why}
        print(f"  {name:<22} {'-':>8}            SKIP  {why}", flush=True)

    # --- the tag through one camera ---------------------------------------------------------------------------
    def look_at_tag(self, cam, frames=6):
        """-> (T_base_tag averaged, list of per-frame (frame dict, T_base_cam, T_cam_tag, corners)) for tag 0."""
        tid, got = self.tags["tags"][0]["id"], []
        for _ in range(frames):
            f = cam.fresh()
            if f is None:
                break
            seen = detect_tags(image_to_gray(f["color"]), self.tags["dictionary"], self.tags["size_m"], f["K"], f["D"],
                               with_corners=True)
            if tid in seen:
                T_base_cam = self.tf(self.base, f["color"].header.frame_id, f["color"].header.stamp)
                got.append((f, T_base_cam, seen[tid][0], seen[tid][1]))
        if len(got) < frames // 2:
            return None, got
        return mean_transform([g[1] @ g[2] for g in got]), got

    @staticmethod
    def tag_center(got):
        """Base-frame ray (origin, unit direction) through the tag CENTER in the image, + its pixel, per frame.
        A single tag's corners come out ~1 px INSIDE the true ones (blur + gamma move black/white edges; the boards'
        checker corners don't have that), so its PnP DISTANCE is biased (~3 % at 0.9 m). The center's direction
        isn't: the shrink is symmetric. So the checks use rays + depth images, never a lone tag's PnP depth."""
        out = []
        for f, T_bc, _, corners in got:
            # where the CENTER projects: the diagonals' crossing (the corners' average is off under perspective)
            h = np.column_stack([corners, np.ones(4)])
            x = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
            uv = x[:2] / x[2]
            r = np.linalg.inv(f["K"]) @ np.array([uv[0], uv[1], 1.0])
            out.append((T_bc[:3, 3], T_bc[:3, :3] @ (r / np.linalg.norm(r)), uv, r))
        return out

    def plane_stats(self, f, T_base_cam, prefix, fit=True, band=0.03):
        """Table points of one depth frame. fit: plane height / tilt in the base frame. Always: the depth SCALE =
        measured depth / depth predicted by intersecting each pixel's ray with the table plane z = 0 (calibrated
        pose + known table)."""
        d = depth_m(f["depth"])
        K = f["K"]
        vs, us = np.mgrid[0:d.shape[0]:4, 0:d.shape[1]:4]
        z = d[::4, ::4]
        ok = np.isfinite(z) & (z > 0.05)
        rays = np.column_stack([(us[ok] - K[0, 2]) / K[0, 0], (vs[ok] - K[1, 2]) / K[1, 1], np.ones(ok.sum())])
        pb = (rays * z[ok][:, None]) @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
        rb = rays @ T_base_cam[:3, :3].T
        sel = (np.abs(pb[:, 2]) < band) & (pb[:, 0] > 0.2) & (pb[:, 0] < 0.95) & (np.abs(pb[:, 1]) < 0.5) & (rb[:, 2] < -0.2)
        if sel.sum() < 200:
            return self.skip(f"{prefix} depth scale", "too little table in view")
        z_pred = -T_base_cam[2, 3] / rb[sel, 2]                  # ray (z = 1 in the camera) hits z = 0 at this depth
        pct = float((np.median(z[ok][sel] / z_pred) - 1.0) * 100)
        if fit:
            P = pb[sel]
            for _ in range(3):                   # least-squares plane z = a x + b y + c, dropping what sticks out
                A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
                coef, *_ = np.linalg.lstsq(A, P[:, 2], rcond=None)
                res = P[:, 2] - A @ coef
                P = P[np.abs(res) < max(0.004, 3 * np.std(res))]
            a, b_, c = coef
            h = (a * 0.5 + c) * 1000             # at the middle of the workspace (0.5, 0)
            self.row(f"{prefix} table height", abs(h), LIMITS["table_mm"], "mm",
                     note=f"z = {h:+.1f} mm at (0.5, 0), {len(P)} points")
            self.row(f"{prefix} table tilt", float(np.degrees(np.arctan(np.hypot(a, b_)))), LIMITS["table_deg"], "deg")
        self.row(f"{prefix} depth scale", abs(pct), LIMITS["depth_pct"], "%",
                 note=f"depth {'longer' if pct > 0 else 'shorter'} than the geometry predicts by {abs(pct):.2f} %")

    # --- checks ---------------------------------------------------------------------------------------------
    def run(self):
        if not self.R.wait_ready():
            raise RuntimeError("robot not ready")
        end = time.time() + 3.0
        while "set" not in self.cal_state and time.time() < end:
            time.sleep(0.05)
        sources = ", ".join(f"{k}: {v['source']}" for k, v in self.cal_state.items() if isinstance(v, dict))
        print(f"calibration set: {self.cal_state.get('set', '?')} ({sources})")
        print("fixed camera (D435i)")
        T_tag_f, got_f = self.look_at_tag(self.fixed)
        self.fixed_rays = self.tag_center(got_f)
        if T_tag_f is None:
            self.skip("tag drift", "the AprilTag isn't visible to the fixed camera (covered?)")
        else:
            self.tag_drift(got_f)
        f = self.fixed.fresh()
        if f is None:
            self.skip("fixed table height", "no fixed-camera frames")
        else:
            self.plane_stats(f, self.tf(self.base, f["color"].header.frame_id), "fixed")
        print("wrist camera (D405)")
        if self.args.no_move:
            self.skip("cross-camera", "--no-move")
        elif T_tag_f is None:
            self.skip("cross-camera", "no tag position from the fixed camera")
        else:
            self.cross_camera(T_tag_f)
        if self.sim:
            print("truth (sim)")
            self.truth()
        bad = [r for r in self.rows if r[4] == "FAIL"]
        print(f"=> {'ALL PASSED' if not bad else 'FAILED: ' + ', '.join(r[0] for r in bad)}  "
              f"({sum(r[4] == 'PASS' for r in self.rows)} pass, {len(bad)} fail, {sum(r[4] == 'SKIP' for r in self.rows)} skip)")
        out = repo_root() / "data" / "calibration" / f"check_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.report, indent=1))
        return 0 if not bad else 1

    def tag_drift(self, got):
        cam = load_fixed_camera()["name"]
        p = calibration_dir() / f"{self.cal_state.get('set', 'real')}_{cam}.yaml"
        ref = (yaml.safe_load(p.read_text()) if p.exists() else {}).get("apriltag_reference", {})
        tid = str(self.tags["tags"][0]["id"])
        if tid not in ref:
            return self.skip("tag drift", f"no reference in {p.name} (tools/calibrate.py fixed / reference)")
        T_ref = make_transform(quat_to_matrix(ref[tid]["quat_xyzw"]), ref[tid]["xyz"])
        px = [np.mean(np.linalg.norm(project(np.linalg.inv(T_bc) @ T_ref, tag_corners(self.tags["size_m"]), f["K"], f["D"])
                                     - corners, axis=1)) for f, T_bc, _, corners in got]
        spread = max(np.linalg.norm((g[1] @ g[2])[:3, 3] - (got[0][1] @ got[0][2])[:3, 3]) for g in got) * 1000
        self.row("tag drift", float(np.median(px)), LIMITS["tag_px"], "px",
                 note=f"(1 px ~ 1.6 mm at 1 m; frame-to-frame spread {spread:.2f} mm)")

    def cross_camera(self, T_tag_f):
        """Put the wrist camera over the tag (a few tries: reach), measure the tag there, compare."""
        T_hand_opt = self.tf(self.wcfg["mount"]["parent"], self.wcfg["frames"]["color_optical"])
        T_hand_tcp = self.tf(self.wcfg["mount"]["parent"], self.robot_cfg["frames"]["tcp"])
        R_ref = top_down_R(self.R.tcp_yaw()) @ T_hand_tcp[:3, :3].T
        cands = eye_in_hand_poses(T_tag_f, T_hand_opt, R_ref, 12, np.random.default_rng(0), dist=(0.22, 0.30),
                                  max_tilt_deg=35.0)
        cands.sort(key=lambda T: np.linalg.norm(T[:2, 3]))          # nearest the robot first: easiest to reach
        home, T_tag_w, got = self.robot_cfg["arm"]["home"], None, []
        try:
            for T in cands[:8]:
                Tt = T @ T_hand_tcp
                if self.R.move_pose(Tt[:3, 3], matrix_to_quat(Tt[:3, :3])) != 1:
                    continue
                time.sleep(0.8)
                T_tag_w, got = self.look_at_tag(self.wrist)
                if T_tag_w is not None:
                    break
        finally:
            self.R.move_joints(home)
        if T_tag_w is None:
            return self.skip("cross-camera", "the wrist camera couldn't get a view of the tag (reach?)")
        # wrist: the tag center as a 3-D point = its ray x the D405 depth there (sub-mm at 25 cm)
        pts = []
        for (o, dvec, uv, r), (f, *_rest) in zip(self.tag_center(got), got):
            d = depth_m(f["depth"])
            u, v = np.round(uv).astype(int)
            patch = d[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3]
            patch = patch[np.isfinite(patch) & (patch > 0.05)]
            if patch.size:
                pts.append(o + dvec * float(np.median(patch)) * np.linalg.norm(r))
        if not pts:
            return self.skip("cross-camera", "no wrist depth at the tag")
        p_w = np.median(pts, axis=0)
        # fixed: the ray through the tag center; the wrist's point must lie ON it (distance perpendicular to it)
        o, dvec = np.mean([c[0] for c in self.fixed_rays], axis=0), np.mean([c[1] for c in self.fixed_rays], axis=0)
        dvec /= np.linalg.norm(dvec)
        off = (p_w - o) - dvec * ((p_w - o) @ dvec)
        self.row("cross-camera", float(np.linalg.norm(off) * 1000), LIMITS["cross_mm"], "mm",
                 note=f"the wrist's tag point is {np.linalg.norm(off) * 1000:.1f} mm off the fixed camera's ray "
                      f"(x, y, z {np.round(off * 1000, 1).tolist()})")
        f = got[-1][0]
        self.plane_stats(f, got[-1][1], "wrist", fit=False)

    def truth(self):
        t = self.bench.sim.call("camera_truth")
        cam = load_fixed_camera()
        pairs = [("fixed camera vs truth", self.tf(self.base, cam["frames"]["color_optical"]), np.array(t["cameras"]["camera"]))]
        if "wrist_in_parent" in t:
            pairs.append(("wrist camera vs truth", self.tf(self.wcfg["mount"]["parent"], self.wcfg["frames"]["color_optical"]),
                          np.array(t["wrist_in_parent"])))
        for name, T_tf, T_true in pairs:
            mm, deg = pose_error(T_tf, T_true)
            self.row(name, mm, LIMITS["truth_mm"], "mm", ok=mm <= LIMITS["truth_mm"] and deg <= LIMITS["truth_deg"],
                     note=f"{deg:.3f} deg (limit {LIMITS['truth_deg']}); profile {t.get('profile')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-move", action="store_true")
    args = ap.parse_args()
    rclpy.init()
    code = Checker(args).run()
    rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
