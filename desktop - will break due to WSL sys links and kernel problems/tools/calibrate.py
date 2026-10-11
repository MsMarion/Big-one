"""Hand-eye calibration: where EXACTLY is each camera relative to the robot? (docs/learning/16_calibration.md)

    python3 /repo/tools/calibrate.py fixed     D435i on its stand: the gripper shows it a ChArUco board (eye-to-hand)
    python3 /repo/tools/calibrate.py wrist     D405 on the hand: it looks at the board on the table (eye-in-hand)
    python3 /repo/tools/calibrate.py reference re-record where the fixed camera sees the AprilTag (done after every
                                               fixed calibration; the drift monitor compares against it)
    python3 /repo/tools/calibrate.py locate    WHERE is the fixed camera at all? (first setup, or after moving it a
                                               lot): from desk targets at known spots; then run `fixed`
        --poses 14     how many good views to collect        --seed 0      which random poses
        --dry-run      solve + report, don't write / apply    --real        no sim: you mount / lay out the boards
        --resolve DIR  re-solve the samples saved by an earlier run (no robot needed)

How it works (pnp_core/calibration.py has the math):
  1. the robot visits ~15 poses chosen so the board is in view and the ROTATIONS differ;
  2. at each one: robot kinematics give T_base_hand (TF), the camera gives T_cam_board (ChArUco + solvePnP);
  3. cv2.calibrateHandEye finds the one transform that makes the "constant" side constant over all poses
     (fixed: the board in the hand; wrist: the board on the table). Several solvers run; the most consistent wins;
  4. the residual (how much the constant side still varies) says how good it is; in the sim we also know the
     TRUE camera pose (sim command camera_truth), so the error is measured, not guessed;
  5. config/calibration/<set>_<camera>.yaml is written (with a history of earlier results) and
     /calibration/reload makes camera_tf_node publish it: every node uses the new pose immediately.
Every view is saved to data/calibration/<camera>_<time>/ (images + samples.json): look at them; --resolve re-solves.
"""

import argparse
import datetime
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402

from pnp_core.calibration import (board_center_T, board_size_m, detect_board, detect_tags, flat_tag_T,  # noqa: E402
                                  survey_tags, tag_layout,
                                  joint_space_board_views, joint_space_wrist_views,
                                  eye_in_hand_poses, image_to_gray, mean_transform,
                                  eye_to_hand_poses, mount_T, pose_error, refine_hand_eye, residuals,
                                  rotation_spread_deg, solve_hand_eye, tag_corners)
from pnp_core.catalog import calibration_dir, calibration_targets_path, load_fixed_camera, load_robot, load_yaml, repo_root  # noqa: E402
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix  # noqa: E402
from pnp_core.object_grasp import top_down_R  # noqa: E402
from pnp_core.wrist import wrist_camera  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402

METHODS = {"park": cv2.CALIB_HAND_EYE_PARK, "tsai": cv2.CALIB_HAND_EYE_TSAI, "horaud": cv2.CALIB_HAND_EYE_HORAUD,
           "andreff": cv2.CALIB_HAND_EYE_ANDREFF, "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS}
GOOD_PX = 1.0     # refined reprojection error below this (rms over all corners of all views): a good calibration


GEOMETRY = ("dictionary", "squares_x", "squares_y", "square_m", "marker_m", "first_id")


def camera_setup(which: str, board: str | None = None) -> dict:
    """Everything that differs between the two calibrations. board: another target's PATTERN for the wrist
    calibration (e.g. "hand_board": the one printed; it lies on the desk like the table board, with the table board's
    view settings)."""
    robot = load_robot()
    targets = load_yaml(calibration_targets_path())
    if which == "fixed":
        cam, topics = load_fixed_camera(), load_yaml("config/interfaces.yaml")["topics"]
        return dict(cam=cam, mode="eye_to_hand", parent=robot["frames"]["base"], board_key="hand_board",
                    board=targets["hand_board"], image=topics["color_image"]["name"],
                    info=topics["color_info"]["name"], truth_key="camera")
    cam = wrist_camera(robot)
    tb = dict(targets["table_board"])
    if board and board != "table_board":
        tb.update({k: targets[board][k] for k in GEOMETRY})
    return dict(cam=cam, mode="eye_in_hand", parent=cam["mount"]["parent"], board_key=board or "table_board",
                board=tb, image=cam["topics"]["color_image"], info=cam["topics"]["color_info"],
                truth_key="wrist_in_parent")


def to_T(tf) -> np.ndarray:
    t, q = tf.translation, tf.rotation
    return make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])


def T_dict(T) -> dict:
    return {"xyz": [round(float(v), 6) for v in T[:3, 3]], "quat_xyzw": [round(float(v), 7) for v in matrix_to_quat(T[:3, :3])]}


def in_view(T_cam_board_c, board_cfg, K, size, margin_px=25) -> bool:
    """Would the whole board (corners of its center frame, +- half size) project inside the image?"""
    w, h = board_size_m(board_cfg)
    pts = np.array([[sx * w / 2, sy * h / 2, 0, 1] for sx in (-1, 1) for sy in (-1, 1)]) @ T_cam_board_c.T
    if np.any(pts[:, 2] < 0.05):
        return False
    uv = (pts[:, :3] @ K.T)[:, :2] / pts[:, 2:3]
    return bool(np.all((uv > margin_px) & (uv < np.array(size) - margin_px)))


def nn_order(Ts: list) -> list:
    """Visit poses nearest-neighbor first (shorter travel between views)."""
    left, out = list(range(len(Ts))), []
    cur = np.zeros(3) + [0.3, 0.0, 0.5]
    while left:
        i = min(left, key=lambda k: np.linalg.norm(Ts[k][:3, 3] - cur))
        out.append(Ts[i])
        cur = Ts[i][:3, 3]
        left.remove(i)
    return out


class Calibrator:
    def __init__(self, which, args):
        self.which, self.args = which, args
        self.s = camera_setup(which, getattr(args, "board", None))
        # the "hand" of the hand-eye problem: the link the board is clamped to / the wrist camera sits on
        self.hand = self.s["board"]["mount"]["parent"] if self.s["mode"] == "eye_to_hand" else self.s["parent"]
        self.robot_cfg = load_robot()
        self.node = Node("pnp_calibrate")
        ex = MultiThreadedExecutor(num_threads=4)
        ex.add_node(self.node)
        threading.Thread(target=ex.spin, daemon=True).start()
        self.R = RobotInterface(self.node, self.robot_cfg)
        self.frames = {"n": 0}
        self.node.create_subscription(Image, self.s["image"], self._img, qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo, self.s["info"], lambda m: self.frames.__setitem__("info", m),
                                      qos_profile_sensor_data)
        self.freeze_cli = self.node.create_client(SetBool, "/perception/map_freeze")
        self.reload_cli = self.node.create_client(Trigger, "/calibration/reload")
        self.cal_state = {}          # camera_tf_node's /calibration/state: which calibration SET is active
        self.node.create_subscription(String, "/calibration/state", lambda m: setattr(self, "cal_state", json.loads(m.data)),
                                      QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                                 reliability=ReliabilityPolicy.RELIABLE))
        self.bench = None if args.real else Bench(name="pnp_calibrate_sim")
        self.out_dir = repo_root() / "data" / "calibration" / \
            f"{self.s['cam']['name']}_{datetime.datetime.now():%Y%m%d_%H%M%S}"

    def _img(self, m):
        self.frames["img"] = m
        self.frames["n"] += 1

    def fresh_frame(self, skip=3, timeout=5.0):
        """A frame captured AFTER the arm stopped (skip a few: the pipeline holds older ones)."""
        n0, end = self.frames["n"], time.time() + timeout
        while (self.frames["n"] < n0 + skip or "info" not in self.frames) and time.time() < end:
            time.sleep(0.02)
        return (self.frames.get("img"), self.frames.get("info")) if self.frames["n"] >= n0 + skip else (None, None)

    def tf(self, parent, child, stamp=None, timeout=5.0) -> np.ndarray:
        """parent -> child at the image stamp (the arm is still, so the latest is the same if that's not buffered)."""
        end = time.time() + timeout
        while True:
            for t in ([Time.from_msg(stamp)] if stamp else []) + [Time()]:
                try:
                    return to_T(self.R.tf.lookup_transform(parent, child, t).transform)
                except Exception:  # noqa: BLE001  (not in the buffer (yet))
                    pass
            if time.time() > end:
                raise RuntimeError(f"no TF {parent} -> {child}")
            time.sleep(0.1)

    def call(self, cli, req, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=2.0):
            return None
        return self.R.wait(cli.call_async(req), timeout)

    # --- the board in the gripper (fixed camera) -------------------------------------------------------------
    def board_in_hand(self, on: bool):
        b = self.s["board"]
        parent = b["mount"]["parent"]
        aco = AttachedCollisionObject(link_name=parent)
        aco.object.id = "calibration_board"
        aco.object.operation = CollisionObject.ADD if on else CollisionObject.REMOVE
        aco.object.header.frame_id = parent
        w, h = board_size_m(b)
        aco.object.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[w + 0.01, h + 0.01, 0.006])]
        T = mount_T(b) @ board_center_T(b)
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, matrix_to_quat(T[:3, :3]))
        aco.object.primitive_poses = [p]
        aco.touch_links = sorted({parent, self.robot_cfg["frames"]["tcp"], *self.robot_cfg.get("hand_links", [])})
        s = PlanningScene(is_diff=True)
        s.robot_state.is_diff = True
        s.robot_state.attached_collision_objects = [aco]
        self.R._apply(s, strict=False)
        if not on:      # MoveIt turns a detached object into a WORLD object: remove that too (it'd sit in the arm)
            w = PlanningScene(is_diff=True)
            w.world.collision_objects = [CollisionObject(id="calibration_board", operation=CollisionObject.REMOVE)]
            self.R._apply(w, strict=False)
        if self.bench:
            self.bench.sim.call("calibration_targets", hand_board=on)
        elif on:
            input(f"REAL: clamp the hand board in the gripper as {calibration_targets_path()} says, press Enter ")

    # --- candidate poses ---------------------------------------------------------------------------------
    def candidates(self, K, size, n):
        rng = np.random.default_rng(self.args.seed)
        s, b = self.s, self.s["board"]
        optical = s["cam"]["frames"]["color_optical"]
        T_hand_tcp = self.tf(self.hand, self.robot_cfg["frames"]["tcp"])
        if s["mode"] == "eye_to_hand":
            T_base_cam = self.tf(s["parent"], optical)          # the current belief: fine for choosing views
            # where the board goes (the cell's targets file; FR3 default: ~0.7 m from its camera, hanging down)
            pz = b.get("poses", {"center": [0.60, 0.20, 0.24], "spread": [0.08, 0.10, 0.06]})
            if pz.get("mode") == "joint":
                # sampled in JOINT space (a 5-joint arm can't take free hand poses): -> joint goals
                if self.R.chain is None:
                    raise RuntimeError("poses.mode joint needs a robot with arm.ik: numeric (its kinematic chain)")
                T_tcp_hand = np.linalg.inv(T_hand_tcp)
                rngs = pz.get("ranges")
                qs = joint_space_board_views(lambda q: self.R.chain.fk(q) @ T_tcp_hand, self.R.chain.lower,
                                             self.R.chain.upper, n * 3, rng, T_base_cam, K, size, mount_T(b), b,
                                             dist=tuple(pz.get("dist", (0.2, 0.45))),
                                             max_view_angle_deg=pz.get("max_view_angle_deg", 45.0), ranges=rngs,
                                             min_visible=float(pz.get("min_visible", 1.0)))
                print(f"{len(qs)} joint configurations show the whole board face-up to the camera", flush=True)
                qs.sort(key=lambda q: float(np.arctan2(*self.R.chain.fk(q)[1::-1, 3])))      # sweep by pan
                return [("q", q) for q in qs]
            raw = eye_to_hand_poses(T_base_cam[:3, 3], mount_T(b), b, n * 12, rng, center=pz["center"],
                                    spread=pz["spread"], tilt_deg=pz.get("tilt_deg", 25.0),
                                    roll_deg=pz.get("roll_deg", 30.0), y_ref=pz.get("y_ref", "down"))
            ok = [T for T in raw if in_view(np.linalg.inv(T_base_cam) @ T @ mount_T(b) @ board_center_T(b), b, K, size)]
        else:
            T_hand_opt = self.tf(s["parent"], optical)
            T_board_c = self.find_table_board()
            pz = b.get("poses", {})
            if pz.get("mode") == "joint":
                # sampled in JOINT space (a 5-joint arm can't look from free poses), like eye_to_hand above
                if self.R.chain is None:
                    raise RuntimeError("poses.mode joint needs a robot with arm.ik: numeric (its kinematic chain)")
                T_tcp_hand = np.linalg.inv(T_hand_tcp)
                qs = joint_space_wrist_views(lambda q: self.R.chain.fk(q) @ T_tcp_hand, self.R.chain.lower,
                                             self.R.chain.upper, n * 3, rng, T_hand_opt, T_board_c, K, size, b,
                                             dist=tuple(pz.get("dist", (0.15, 0.24))),
                                             max_view_angle_deg=pz.get("max_view_angle_deg", 35.0),
                                             fk_tcp=self.R.chain.fk, ranges=pz.get("ranges"),
                                             region_px=pz.get("image_region_px"))
                print(f"{len(qs)} joint configurations show the whole table board to the wrist camera", flush=True)
                qs.sort(key=lambda q: float(np.arctan2(*self.R.chain.fk(q)[1::-1, 3])))      # sweep by pan
                return [("q", q) for q in qs]
            R_ref = top_down_R(self.R.tcp_yaw()) @ T_hand_tcp[:3, :3].T     # TCP top-down -> hand rotation
            raw = eye_in_hand_poses(T_board_c, T_hand_opt, R_ref, n * 12, rng)
            ok = [T for T in raw if in_view(np.linalg.inv(T @ T_hand_opt) @ T_board_c, b, K, size)]
        print(f"{len(ok)} of {len(raw)} candidate poses keep the whole board in view", flush=True)
        return [("pose", T @ T_hand_tcp) for T in nn_order(ok)]             # -> TCP poses (what move_pose takes)

    def find_table_board(self) -> np.ndarray:
        """Where is the table board (center frame, base)? Ask the FIXED camera (cross-camera); else the WRIST camera
        from where the arm is now (its current mount: good to ~1-3 cm, plenty to plan views around: no fixed camera
        on the real desk yet, and the board needn't lie exactly at its configured spot); else the config."""
        b = self.s["board"]
        base = self.robot_cfg["frames"]["base"]
        topics = load_yaml("config/interfaces.yaml")["topics"]
        sources = [("fixed", topics["color_image"]["name"], topics["color_info"]["name"])]
        if self.s["mode"] == "eye_in_hand":
            # the wrist camera FIRST: the fixed one may not be calibrated (moved / remounted: on the new plate it put
            # the board 13 cm off and every planned view was unreachable)
            sources.insert(0, ("wrist", self.s["image"], self.s["info"]))
        for who, img_topic, info_topic in sources:
            got = {}
            subs = [self.node.create_subscription(Image, img_topic, lambda m, g=got: g.__setitem__("img", m),
                                                  qos_profile_sensor_data),
                    self.node.create_subscription(CameraInfo, info_topic, lambda m, g=got: g.__setitem__("info", m),
                                                  qos_profile_sensor_data)]
            end = time.time() + 4.0
            while ("img" not in got or "info" not in got) and time.time() < end:
                time.sleep(0.05)
            for sub in subs:
                self.node.destroy_subscription(sub)
            if "img" in got and "info" in got:
                T_cb, n, err = detect_board(image_to_gray(got["img"]), b, np.array(got["info"].k).reshape(3, 3),
                                            got["info"].d)
                if T_cb is not None:
                    T = self.tf(base, got["img"].header.frame_id, got["img"].header.stamp) @ T_cb @ board_center_T(b)
                    print(f"table board seen by the {who} camera at {np.round(T[:3, 3], 3).tolist()} ({n} corners)")
                    return T
        sp = b["sim_pose"]
        print("no camera sees the table board: using its configured position")
        c, s_ = np.cos(np.radians(sp["yaw_deg"])), np.sin(np.radians(sp["yaw_deg"]))
        return make_transform(np.array([[c, s_, 0], [s_, -c, 0], [0, 0, -1.0]]), [sp["xy"][0], sp["xy"][1], 0.0])

    # --- collect ------------------------------------------------------------------------------------------
    def collect(self):
        s, b = self.s, self.s["board"]
        n_corners = (b["squares_x"] - 1) * (b["squares_y"] - 1)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _, info = self.fresh_frame()
        if info is None:
            raise RuntimeError(f"no frames on {s['image']}")
        K, size = np.array(info.k).reshape(3, 3), (info.width, info.height)
        self.K, self.D = K, np.array(info.d, float) if len(info.d) else np.zeros(5)
        samples = []
        tcp_poses = self.candidates(K, size, self.args.poses)
        for i, (kind, T) in enumerate(tcp_poses):
            if len(samples) >= self.args.poses:
                break
            code = self.R.move_joints(T) if kind == "q" else self.R.move_pose(T[:3, 3], matrix_to_quat(T[:3, :3]))
            if code != 1:
                print(f"  pose {i:2d}: not reachable (MoveIt {code}), skipped", flush=True)
                continue
            time.sleep(self.args.settle)
            img, info = self.fresh_frame()
            if img is None:
                print(f"  pose {i:2d}: no camera frame", flush=True)
                continue
            g = image_to_gray(img)
            T_cb, n, err, pts = detect_board(g, b, K, self.D, return_points=True)
            T_bh = self.tf(self.robot_cfg["frames"]["base"], self.hand, img.header.stamp)
            cv2.imwrite(str(self.out_dir / f"view_{i:02d}.png"), g)
            if T_cb is None or n < n_corners // 2 or err > 2.0:
                print(f"  pose {i:2d}: board not usable ({n}/{n_corners} corners, {err:.2f} px)", flush=True)
                continue
            samples.append({"T_base_hand": T_bh.tolist(), "T_cam_board": T_cb.tolist(), "corners": n,
                            "reproj_px": err, "view": f"view_{i:02d}.png",
                            "board_pts": pts[0].round(6).tolist(), "pixels": pts[1].round(3).tolist()})
            d = np.linalg.norm(T_cb[:3, 3])
            print(f"  pose {i:2d}: {n}/{n_corners} corners, {err:.2f} px, board at {d:.2f} m  "
                  f"[{len(samples)}/{self.args.poses}]", flush=True)
        (self.out_dir / "samples.json").write_text(json.dumps({"camera": self.which, "K": K.tolist(),
                                                               "D": self.D.tolist(), "samples": samples}))
        return samples

    def run(self):
        s = self.s
        print(f"=== {self.which} camera ({s['cam']['name']}): {s['mode'].replace('_', '-')} with the {s['board_key']}")
        if not self.R.wait_ready():
            raise RuntimeError("robot not ready")
        before = self.tf(s["parent"], s["cam"]["frames"]["color_optical"])      # what the system believes now
        home = self.robot_cfg["arm"]["home"]
        frozen = False
        try:
            self.R.move_joints(home)
            if s["mode"] == "eye_to_hand":
                # the board would be mapped as an obstacle stuck to the hand: hold the obstacle map. And EMPTY it: it
                # was built with the camera pose being calibrated, so it can hold the robot itself where the wrong
                # pose put it (a miscalibrated camera's map put an "obstacle" on the shoulder: every plan refused)
                frozen = bool(getattr(self.call(self.freeze_cli, SetBool.Request(data=True)), "success", False))
                if frozen:
                    self.R.clear_octomap()
                self.board_in_hand(True)
            else:
                # jaws wide open: they're in the wrist view (the SO-101's hide part of its bottom third)
                self.R.gripper(self.robot_cfg["gripper"]["max_width_m"], 5.0)
                if self.bench:
                    # a clear desk: the target (and boxes) could lie on the board and hide its corners
                    self.bench.sim.call("target_away")
                    self.bench.sim.call("clear_obstacles")
                    self.bench.sim.call("calibration_targets", table_board=True)
                else:
                    sp = s["board"]["sim_pose"]
                    input(f"REAL: clear the work area, lay the table board flat, printed side up, at about "
                          f"x={sp['xy'][0]} y={sp['xy'][1]} (base frame), its long side along x; press Enter ")
            time.sleep(0.5)
            samples = self.collect()
        finally:
            self.R.move_joints(home)
            if s["mode"] == "eye_to_hand":
                self.board_in_hand(False)
                if frozen:
                    self.call(self.freeze_cli, SetBool.Request(data=False))
                    self.R.clear_octomap()               # rebuilt from the (new) calibration
            elif self.bench:
                self.bench.sim.call("calibration_targets", table_board=False)
                self.bench.sim.call("reset_target")
        return self.solve_and_save(samples, before)

    # --- solve, judge, save ------------------------------------------------------------------------------
    def solve_and_save(self, samples, before=None):
        s = self.s
        print(f"saved views + samples: {self.out_dir}")
        if len(samples) < 6:
            print(f"only {len(samples)} usable views (need >= 6): not solving")
            return 1
        A = [np.array(x["T_base_hand"]) for x in samples]
        B = [np.array(x["T_cam_board"]) for x in samples]
        print(f"{len(samples)} views, hand rotations span {rotation_spread_deg(A):.0f} deg, "
              f"mean reprojection {np.mean([x['reproj_px'] for x in samples]):.2f} px")
        truth = self.truth()

        def judge(X):
            return " {:13.2f} {:6.3f}".format(*pose_error(X, truth)) if truth is not None else ""
        # 1) closed form, all five classic solvers (each weighs rotation/translation differently)
        results = {}
        print(f"{'method':<16} {'residual mm':>11} {'deg':>6}" + (f" {'TRUE err mm':>13} {'deg':>6}" if truth is not None else ""))
        for name, m in METHODS.items():
            try:
                X = solve_hand_eye(A, B, s["mode"], m)
            except cv2.error:
                continue
            r = residuals(A, B, X, s["mode"])
            results[name] = (X, r)
            print(f"{name:<16} {r['pos_mm_rms']:11.2f} {r['rot_deg_rms']:6.3f}" + judge(X))
        # the CONSENSUS method, not the one with the lowest residual of its own: the residual is self-consistency
        # (2026-10-11, sim: park+refine had the best residual, 0.39 mm, and the worst TRUE error, 3.43 mm; plain park
        # 0.81 / 0.57). The method closest to the others (sum over the nearer half: an outlier like tsai can't pull it)
        names = list(results)

        def dist(p, q):
            e = pose_error(results[p][0], results[q][0])
            return e[0] + 10.0 * e[1]                  # 0.1 deg ~ 1 mm
        half = max(1, (len(names) - 1) // 2 + 1)
        best = min(names, key=lambda k: sum(sorted(dist(k, o) for o in names if o != k)[:half]))
        spread = max((dist(p, q) for p in names for q in names if p < q and dist(best, p) < 5 and dist(best, q) < 5),
                     default=0.0)
        print(f"consensus: {best} (the methods near it agree within {spread:.2f} mm-equivalent)")
        X_cons, C_cons, cons = results[best][0], results[best][1]["constant"], best
        # 2) refine the most consistent one in PIXELS (camera + constant side together); drop outlier views once
        obs = [(np.array(x["board_pts"]), np.array(x["pixels"])) for x in samples]
        keep = list(range(len(samples)))
        X, C = results[best][0], results[best][1]["constant"]
        for _ in range(2):
            X, C, rms, per_view = refine_hand_eye([A[i] for i in keep], [obs[i] for i in keep], X, C, s["mode"],
                                                  self.K, self.D)
            bad = [keep[j] for j, e in enumerate(per_view) if e > max(1.0, 3 * np.median(per_view))]
            if not bad or len(keep) - len(bad) < 6:
                break
            print(f"dropping outlier views {[samples[i]['view'] for i in bad]}")
            keep = [i for i in keep if i not in bad]
        A, B = [A[i] for i in keep], [B[i] for i in keep]
        r = residuals(A, B, X, s["mode"])
        print(f"{best + '+refine':<16} {r['pos_mm_rms']:11.2f} {r['rot_deg_rms']:6.3f}" + judge(X)
              + f"   reprojection {rms:.2f} px")
        moved = pose_error(X, X_cons)
        refined = moved[0] <= 1.0 and moved[1] <= 0.2
        if not refined:      # the pixel refinement left the consensus: it fitted something else (e.g. the arm's errors)
            print(f"refinement moved {moved[0]:.2f} mm / {moved[1]:.3f} deg away from the consensus ({cons}): kept {cons}")
            X, C = X_cons, C_cons
            r = residuals(A, B, X, s["mode"])
        good = rms <= GOOD_PX
        print(f"-> camera at {np.round(X[:3, 3], 4).tolist()} in {s['parent']}: reprojection {rms:.2f} px over "
              f"{len(keep)} views ({'GOOD' if good else 'POOR: check the views'})")
        best = best + "+refine" if refined else best
        # the constant side vs what we configured: the board in the hand (fixed) / on the table (wrist)
        entry = {"camera": s["cam"]["name"], "parent": s["parent"], "optical_frame": s["cam"]["frames"]["color_optical"],
                 "T_parent_optical": T_dict(X), "date": datetime.datetime.now().isoformat(timespec="seconds"),
                 "method": best, "views": len(keep), "rotation_span_deg": round(rotation_spread_deg(A), 1),
                 "reprojection_px": round(rms, 3),
                 "residual": {"pos_mm_rms": round(r["pos_mm_rms"], 3), "rot_deg_rms": round(r["rot_deg_rms"], 4)},
                 "board": s["board_key"], "board_found": T_dict(C), "samples": str(self.out_dir.relative_to(repo_root()))}
        if s["mode"] == "eye_to_hand":
            pe = pose_error(C, mount_T(s["board"]))
            print(f"board in the hand: {pe[0]:.1f} mm / {pe[1]:.2f} deg from its configured mount "
                  f"(a real clamp is never exact: that's fine, it's solved for)")
        if truth is not None:
            e = pose_error(X, truth)
            entry["sim_truth_error"] = {"pos_mm": round(e[0], 3), "rot_deg": round(e[1], 4)}
            entry["sim_profile"] = self.truth_profile
            line = f"TRUE error (sim): {e[0]:.2f} mm / {e[1]:.3f} deg"
            if before is not None:
                eb = pose_error(before, truth)
                entry["before"] = {"pos_mm": round(eb[0], 3), "rot_deg": round(eb[1], 4)}
                line += f"   (before this calibration: {eb[0]:.2f} mm / {eb[1]:.3f} deg)"
            print(line)
        if self.args.dry_run:
            print("dry run: nothing written")
            return 0 if good else 2
        if not good and not self.args.force:
            print("residual too large: NOT written (look at the saved views; --force writes anyway)")
            return 2
        self.write(entry)
        if self.which == "fixed" and self.args.resolve is None:
            self.record_reference()             # the camera just changed: so does where it sees the tag
        return 0 if good else 2

    def locate(self):
        """The fixed camera's pose FROM SCRATCH, for a camera that isn't where the scene file says (`fixed` picks its
        views from the believed pose: far off, the board isn't in them). Everything here lies on the desk at a KNOWN
        spot: the AprilTag(s) at their configured xy and the table board at its sim_pose (sim: shown; real: you lay
        it there). Every detected corner is a 3-D point (base frame) seen at a pixel: one solvePnP over all of them
        -> the camera pose. Coarse (as good as the targets are placed); written as the calibration (method locate)
        so `fixed` can take it from here. Also saves an image of what it found with the work areas drawn in, and
        checks that the work areas are in view."""
        import cv2
        from pnp_core.catalog import load_scene
        targets = load_yaml(calibration_targets_path())
        path = self.bench.get_pick_param("scene") if self.bench else None
        scene = load_yaml(path) if path else load_scene()
        table_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
        tb = targets.get("table_board")
        if self.bench:
            self.bench.sim.call("target_away")
            self.bench.sim.call("clear_obstacles")
            if tb:
                self.bench.sim.call("calibration_targets", table_board=True)
        elif self.args.via_wrist:
            # no measured spot on the desk: the WRIST camera (hand-eye calibrated) says where the board lies
            tb = camera_setup("wrist", self.args.board)["board"]
            T_base_board = self.board_from_wrist(tb)
            if T_base_board is None:
                return 1
        elif tb:
            ans = input(f"REAL: lay the table board flat at x={tb['sim_pose']['xy'][0]} y={tb['sim_pose']['xy'][1]} "
                        f"(long side along x) for a better result, then y + Enter (Enter alone: tags only) ")
            if ans.strip().lower() != "y":
                tb = None
        if not self.args.no_move:
            self.R.move_joints(self.robot_cfg["arm"]["home"])     # out of the view
            time.sleep(1.5)
        self.frames.pop("img", None)
        end = time.time() + 5.0
        while ("img" not in self.frames or "info" not in self.frames) and time.time() < end:
            time.sleep(0.05)
        if "img" not in self.frames:
            print("no image from the fixed camera")
            return 1
        img, info = self.frames["img"], self.frames["info"]
        K = np.array(info.k).reshape(3, 3)
        D = np.array(info.d, float) if len(info.d) else np.zeros(5)
        gray = image_to_gray(img)
        P3, P2, found = [], [], []

        flat = flat_tag_T                # a target lying printed side up: x along yaw, y "down" the print, z into it
        a = targets["apriltags"]
        tags = detect_tags(gray, a["dictionary"], a["size_m"], K, D, with_corners=True)
        survey_path = calibration_dir() / f"{self.active_set()}_tag_layout.yaml"
        survey = yaml.safe_load(survey_path.read_text()) if survey_path.exists() else None
        layout = tag_layout(a, survey)          # where the tags lie: surveyed (locate --survey) or as planned
        if survey:
            print(f"tag positions: surveyed {survey.get('date')} ({survey_path.name})")
        for tid, (x, y, yaw) in layout.items():
            if tid in tags:
                T = flat((x, y), yaw, table_z)
                P3 += list(tag_corners(a["size_m"]) @ T[:3, :3].T + T[:3, 3])
                P2 += list(tags[tid][1])
                found.append(f"AprilTag {tid}")
        if tb:
            T_cb, n, _, pts = detect_board(gray, tb, K, D, return_points=True)
            if T_cb is not None:
                T = T_base_board if self.args.via_wrist else \
                    flat(tb["sim_pose"]["xy"], tb["sim_pose"]["yaw_deg"], table_z) @ np.linalg.inv(board_center_T(tb))
                P3 += list(pts[0] @ T[:3, :3].T + T[:3, 3])
                P2 += list(pts[1])
                found.append(f"{'board placed by the wrist camera' if self.args.via_wrist else 'table board'} "
                             f"({n} corners)")
        if self.bench:
            if tb:
                self.bench.sim.call("calibration_targets", table_board=False)
            self.bench.sim.call("reset_target")
        if len(P3) < 4:
            print(f"found {found or 'nothing'}: need the AprilTag and/or the table board in the camera's view")
            return 2
        P3, P2 = np.array(P3, np.float64), np.array(P2, np.float64)
        flag = cv2.SOLVEPNP_ITERATIVE if len(P3) > 4 else cv2.SOLVEPNP_IPPE
        ok, rvec, tvec = cv2.solvePnP(P3, P2, K, D, flags=flag)
        proj, _ = cv2.projectPoints(P3, rvec, tvec, K, D)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - P2) ** 2, axis=1))))
        X = np.linalg.inv(make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel()))        # base -> optical
        if self.args.survey:
            # hand-placed tags: where does each REALLY lie, relative to the others? (pnp_core/calibration.py
            # survey_tags). A PLACEMENT CHECK: the camera still comes from the plan above -- the survey's extra freedom
            # also bends the layout around whatever the camera model gets wrong (sim: ~1 px, -> 6 mm on the camera,
            # vs 0.3 mm from the plan). Noise floor ~2 mm (sim, exact layout): a tag > survey_warn_mm off is misplaced:
            # move it to its planned spot, or (no room there) keep where it is: --keep-survey.
            obs = {tid: tags[tid][1] for tid in layout if tid in tags}
            if len(obs) < 4:
                print(f"survey: {len(obs)} tags in view, need >= 4")
                return 2
            X_s, lay, rms_s, size_app = survey_tags(obs, a["size_m"], K, D, layout, table_z, X)
            print(f"survey: {len(obs)} tags, reprojection {rms_s:.2f} px, apparent tag size {size_app * 1000:.2f} mm "
                  f"(printed {a['size_m'] * 1000:.1f}); each tag vs the plan:")
            off = []
            for tid in sorted(lay):
                d = (np.array(lay[tid][:2]) - layout[tid][:2]) * 1000
                dyaw = lay[tid][2] - layout[tid][2]
                bad = np.hypot(*d) > self.args.survey_warn_mm or abs(dyaw) > 3.0
                off += [tid] if bad else []
                print(f"   tag {tid}: x {lay[tid][0] * 100:6.1f} cm  y {lay[tid][1] * 100:6.1f} cm  yaw {lay[tid][2]:+5.1f}"
                      f"   off the plan {d[0]:+5.1f} / {d[1]:+5.1f} mm ({d[0] / 25.4:+.2f} / {d[1] / 25.4:+.2f} in), "
                      f"{dyaw:+.1f} deg" + ("   <- MOVE IT (or --keep-survey)" if bad else ""))
            print(f"   {'all within ' + str(self.args.survey_warn_mm) + ' mm / 3 deg of the plan' if not off else f'{len(off)} misplaced: {off}'}"
                  f" (x forward from the base, y to its left)")
            if self.args.keep_survey:
                # the misplaced tags' surveyed positions become the layout (the others stay as planned); rerun locate
                survey = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "reprojection_px": round(rms_s, 3),
                          "apparent_size_m": round(float(size_app), 5),
                          "tags": {int(t): {"xy": [round(float(lay[t][0]), 5), round(float(lay[t][1]), 5)],
                                            "yaw_deg": round(float(lay[t][2]), 3)} for t in off}}
                if not self.args.dry_run:
                    survey_path.write_text(
                        "# AprilTags NOT at their planned spot, where they really lie (tools/calibrate.py locate --survey\n"
                        "# --keep-survey). calibrate.py locate and tools/wrist_tag_check.py use these instead of the plan.\n"
                        + yaml.safe_dump(survey, sort_keys=False, default_flow_style=None))
                    print(f"written: {survey_path.relative_to(repo_root())} ({len(off)} tags): run locate again (no --survey)")
                return 0
        p, z = X[:3, 3], X[:3, 2]
        hit = p + z * (table_z - p[2]) / z[2] if z[2] < -1e-6 else None
        tilt = float(np.degrees(np.arccos(np.clip(-z[2], -1, 1))))
        print(f"found: {', '.join(found)} ({len(P3)} points), reprojection {rms:.2f} px")
        where = (f", looking at desk point {np.round(hit[:2], 3).tolist()} ({np.linalg.norm(p - hit):.2f} m away)"
                 if hit is not None else ", NOT looking at the desk")
        print(f"-> camera at {np.round(p, 3).tolist()} m (base frame), tilted {tilt:.1f} deg from straight down{where}")
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for (u, v), (pu, pv) in zip(P2, proj.reshape(-1, 2)):
            cv2.circle(vis, (int(u), int(v)), 4, (0, 255, 0), 1)
            cv2.drawMarker(vis, (int(pu), int(pv)), (0, 0, 255), cv2.MARKER_CROSS, 6, 1)
        Xi = np.linalg.inv(X)
        for name, reg, col in (("pick area", scene["target"]["spawn_region"], (255, 128, 0)),
                               ("place area", scene["goal"].get("region", scene["target"]["spawn_region"]),
                                (0, 200, 255))):
            c3 = np.array([[x, y, table_z] for x, y in ((reg["x"][0], reg["y"][0]), (reg["x"][1], reg["y"][0]),
                                                         (reg["x"][1], reg["y"][1]), (reg["x"][0], reg["y"][1]))])
            pc = c3 @ Xi[:3, :3].T + Xi[:3, 3]
            uv = (pc @ K.T)[:, :2] / pc[:, 2:3]
            inside = bool(np.all(pc[:, 2] > 0) and np.all((uv >= 0) & (uv < [img.width, img.height])))
            print(f"   {name}: {'fully in view' if inside else 'NOT fully in view'}")
            cv2.polylines(vis, [uv.astype(np.int32)], True, col, 2)
            cv2.putText(vis, name, (int(uv[0, 0]) + 4, int(uv[0, 1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.out_dir / "locate.png"), vis)
        print(f"image (green: detected, red: reprojected, boxes: work areas): "
              f"{(self.out_dir / 'locate.png').relative_to(repo_root())}")
        truth = self.truth()
        entry = {"camera": self.s["cam"]["name"], "parent": self.robot_cfg["frames"]["base"],
                 "optical_frame": self.s["cam"]["frames"]["color_optical"], "T_parent_optical": T_dict(X),
                 "date": datetime.datetime.now().isoformat(timespec="seconds"), "method": "locate (desk targets)",
                 "points": int(len(P3)), "reprojection_px": round(rms, 3), "targets": found}
        if truth is not None:
            e = pose_error(X, truth)
            entry["sim_truth_error"] = {"pos_mm": round(e[0], 3), "rot_deg": round(e[1], 4)}
            entry["sim_profile"] = self.truth_profile
            print(f"TRUE error (sim): {e[0]:.1f} mm / {e[1]:.2f} deg")
        if self.args.dry_run:
            print("dry run: nothing written")
            return 0
        self.write(entry)
        print("next: python3 /repo/tools/calibrate.py fixed" + (" --real" if self.args.real else "")
              + "   (the precise hand-eye calibration, starting from this pose)")
        return 0

    def board_from_wrist(self, b, look=(0.0, 0.0, 0.0, np.radians(80), 0.0), frames=8):
        """The board lying on the desk, in the base frame, as the wrist camera sees it from the look pose (averaged
        over a few frames). -> T_base_board (OpenCV board frame: corner origin), or None."""
        w = camera_setup("wrist", self.args.board)
        got = {"n": 0}

        def img(m):
            got["img"] = m
            got["n"] += 1
        subs = [self.node.create_subscription(Image, w["image"], img, qos_profile_sensor_data),
                self.node.create_subscription(CameraInfo, w["info"], lambda m: got.__setitem__("info", m),
                                              qos_profile_sensor_data)]
        self.R.move_joints(list(look), ptp=True)
        time.sleep(1.5)
        Ts, base = [], self.robot_cfg["frames"]["base"]
        for _ in range(frames):
            n0, end = got["n"], time.time() + 3.0
            while (got["n"] < n0 + 3 or "info" not in got) and time.time() < end:
                time.sleep(0.02)
            if got["n"] < n0 + 3:
                break
            m, info = got["img"], got["info"]
            T_cb, n, err = detect_board(image_to_gray(m), b, np.array(info.k).reshape(3, 3), np.array(info.d, float))
            if T_cb is not None:
                Ts.append(self.tf(base, m.header.frame_id, m.header.stamp) @ T_cb)
        for s in subs:
            self.node.destroy_subscription(s)
        if len(Ts) < frames // 2:
            print(f"the wrist camera saw the board in {len(Ts)}/{frames} frames from the look pose: lay it flat in "
                  f"front of the robot (~20 cm out) and try again")
            return None
        T = mean_transform(Ts)
        spread = max(np.linalg.norm(X[:3, 3] - T[:3, 3]) for X in Ts) * 1000
        c = (T @ board_center_T(b))[:3, 3]
        print(f"wrist camera: board center at {np.round(c, 4).tolist()} m (base), {len(Ts)} frames, spread "
              f"{spread:.1f} mm")
        return T

    def record_reference(self, frames=10):
        """Where does the (calibrated) fixed camera see the AprilTag(s) now? Stored in its calibration file as the
        drift monitor's reference. The tag must not move afterwards (a moved tag: run this again)."""
        tcfg = load_yaml(calibration_targets_path())["apriltags"]
        cal_set = self.active_set()
        path = calibration_dir() / f"{cal_set}_{self.s['cam']['name']}.yaml"
        if self.which != "fixed" or not path.exists():
            print(f"no {path.name}: calibrate the fixed camera first (tools/calibrate.py fixed)")
            return 1
        seen = {}
        for _ in range(frames):
            img, info = self.fresh_frame(skip=2)
            if img is None:
                break
            K, D = np.array(info.k).reshape(3, 3), np.array(info.d, float) if len(info.d) else np.zeros(5)
            T_base_cam = self.tf(self.robot_cfg["frames"]["base"], img.header.frame_id, img.header.stamp)
            for tid, T in detect_tags(image_to_gray(img), tcfg["dictionary"], tcfg["size_m"], K, D).items():
                seen.setdefault(tid, []).append(T_base_cam @ T)
        ref = {tid: mean_transform(Ts) for tid, Ts in seen.items() if len(Ts) >= frames // 2}
        if not ref:
            print("no AprilTag seen by the fixed camera: no reference recorded (is it covered?)")
            return 1
        for tid, T in ref.items():
            spread = max(np.linalg.norm(X[:3, 3] - T[:3, 3]) for X in seen[tid]) * 1000
            line = f"AprilTag {tid}: at {np.round(T[:3, 3], 4).tolist()}, {len(seen[tid])} frames (spread {spread:.2f} mm)"
            truth = [t for t in tcfg["tags"] if t["id"] == tid]
            if self.bench and truth:              # sim: where the tag REALLY is -> an independent accuracy check
                c, s_ = np.cos(np.radians(truth[0]["sim_yaw_deg"])), np.sin(np.radians(truth[0]["sim_yaw_deg"]))
                from pnp_core.catalog import load_scene
                tb_ = load_scene()["table"]                # the tag lies 0.8 mm above THIS cell's board (not z = 0)
                T_true = make_transform(np.array([[c, s_, 0], [s_, -c, 0], [0, 0, -1.0]]),
                                        [truth[0]["sim_xy"][0], truth[0]["sim_xy"][1],
                                         tb_["center"][2] + tb_["size"][2] / 2 + 0.0008])
                cam = self.tf(self.robot_cfg["frames"]["base"], self.s["cam"]["frames"]["color_optical"])[:3, 3]
                ray = (T_true[:3, 3] - cam) / np.linalg.norm(T_true[:3, 3] - cam)
                e = T[:3, 3] - T_true[:3, 3]
                along = float(e @ ray)
                side = float(np.linalg.norm(e - ray * along))
                line += (f"; vs TRUE: {side * 1000:.1f} mm sideways, {along * 1000:+.1f} mm along the view (a lone tag's "
                         f"distance is biased: blur moves its edges; the drift monitor compares against itself)")
            print(line)
        c = yaml.safe_load(path.read_text())
        c["apriltag_reference"] = {str(k): T_dict(T) for k, T in ref.items()}
        c["apriltag_reference_date"] = datetime.datetime.now().isoformat(timespec="seconds")
        head = "".join(f"{line}\n" for line in path.read_text().splitlines() if line.startswith("#"))
        path.write_text(head + yaml.safe_dump(c, sort_keys=False, default_flow_style=None, width=120))
        self.call(self.reload_cli, Trigger.Request())            # -> /calibration/state -> the monitor reloads
        print(f"reference recorded in {path.relative_to(repo_root())}")
        return 0

    def active_set(self) -> str:
        """The calibration set camera_tf_node uses now (sim: sim_<reality profile>; hardware: real)."""
        end = time.time() + 3.0
        while "set" not in self.cal_state and time.time() < end:
            time.sleep(0.05)
        return self.cal_state.get("set") or ("real" if self.args.real else "sim_ideal")

    def truth(self):
        self.truth_profile = None
        if not self.bench:
            return None
        t = self.bench.sim.call("camera_truth")
        self.truth_profile = t.get("profile")
        v = t.get(self.s["truth_key"]) if self.s["truth_key"] != "camera" else t["cameras"].get("camera")
        return None if v is None else np.array(v)

    def write(self, entry):
        cal_set = self.active_set()
        path = calibration_dir() / f"{cal_set}_{entry['camera']}.yaml"
        old = yaml.safe_load(path.read_text()) if path.exists() else {}
        hist = old.pop("history", []) if old else []
        if old:
            hist = [{k: old[k] for k in ("date", "T_parent_optical", "method", "reprojection_px", "sim_truth_error",
                                         "sim_profile") if k in old}] + hist
        entry["history"] = hist[:10]
        head = (f"# Hand-eye calibration of the {entry['camera']} (tools/calibrate.py {self.which}). camera_tf_node "
                f"publishes\n# T_parent_optical as {entry['parent']} -> {entry['optical_frame']}. history: earlier "
                f"results, newest first.\n")
        path.write_text(head + yaml.safe_dump(entry, sort_keys=False, default_flow_style=None, width=120))
        print(f"written: {path.relative_to(repo_root())}")
        res = self.call(self.reload_cli, Trigger.Request())
        if res is None:
            print("camera_tf_node not running: the new calibration applies at the next bringup")
            return
        time.sleep(1.0)
        now = self.tf(entry["parent"], entry["optical_frame"])
        X = make_transform(quat_to_matrix(entry["T_parent_optical"]["quat_xyzw"]), entry["T_parent_optical"]["xyz"])
        e = pose_error(now, X)
        print(f"applied: TF now {e[0]:.2f} mm / {e[1]:.3f} deg from the result ({res.message})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camera", choices=["fixed", "wrist", "reference", "locate"])
    ap.add_argument("--poses", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=0.8, help="seconds to wait after each move before the photo")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="write even if the residual is poor")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--survey", action="store_true",
                    help="locate: check the hand-placed tags: where each really lies vs the plan (needs >= 4 in view)")
    ap.add_argument("--survey-warn-mm", type=float, default=5.0)   # the user: 3-4 mm is an acceptable placement
    ap.add_argument("--keep-survey", action="store_true",
                    help="locate --survey: keep the misplaced tags where they are (their measured positions -> "
                         "<set>_tag_layout.yaml), then run locate again")
    ap.add_argument("--no-move", action="store_true", help="locate: don't send the arm home first (it's out of view)")
    ap.add_argument("--board", choices=["table_board", "hand_board"],
                    help="wrist: the PATTERN on the desk (default table_board; hand_board = the bigger one, if that's "
                         "what was printed)")
    ap.add_argument("--via-wrist", action="store_true",
                    help="locate --real: the board's spot on the desk comes from the WRIST camera (calibrate it "
                         "first), not from a measured layout; with --board = the pattern lying there")
    ap.add_argument("--resolve", help="a data/calibration/<run> folder: re-solve its samples.json, no robot")
    args = ap.parse_args()
    rclpy.init()
    c = Calibrator("fixed" if args.camera in ("reference", "locate") else args.camera, args)
    if args.camera == "reference":
        code = c.record_reference()
    elif args.camera == "locate":
        code = c.locate()
    elif args.resolve:
        d = Path(args.resolve)
        c.out_dir = d if d.is_absolute() else repo_root() / d
        saved = json.loads((c.out_dir / "samples.json").read_text())
        c.K, c.D = np.array(saved["K"]), np.array(saved["D"])
        code = c.solve_and_save(saved["samples"])
    else:
        code = c.run()
    rclpy.try_shutdown()
    return code


if __name__ == "__main__":
    sys.exit(main())
