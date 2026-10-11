"""Pick & place action server: /pick_place (pnp_interfaces/PickPlace).

State machine (feedback.state reports where it is):

    HOME -> DETECT -> MOVE_PREGRASP -> [LOOK] -> APPROACH -> GRASP -> VERIFY_GRASP -> LIFT   (up to 3 attempts)
         -> MOVE_PREPLACE -> DESCEND -> RELEASE -> RETREAT -> HOME -> VERIFY_PLACE

  * free-space moves: OMPL (collision-aware; the table and the KNOWN obstacles are in the planning scene)
  * "above + straight down" pairs (pre-grasp -> grasp, pre-place -> place) are planned TOGETHER before moving:
    the straight line (Pilz LIN) has no collision avoidance of its own, and whether it's free depends on which
    of the 7-joint arm's many configurations (elbow position) the free-space move ends in (_plan_approach)
  * known obstacles (Phase 5, Layer A): boxes from /scene/known_obstacles (std_msgs/String JSON, published by
    the sim; on a real robot by whatever holds the measured map), mirrored into MoveIt as soon as they change
  * approach / lift / descend / retreat: Pilz LIN (straight, slow, predictable)
  * grasp check: finger width after closing ~ object width -> holding; ~0 -> missed (retry)
  * drop check: width collapses during transport -> DROPPED
  * place check: re-detect after returning home; error to the goal must be < place_tolerance_m

Where the object pose comes from (DETECT, param grasp_source):
  pose      the 6-D pose node (FoundationPose, /perception/pose_status): position AND orientation; the grasp is
            computed in the object's own frame (pnp_core/object_grasp.py), so a tilted / lying object is handled
  detector  YOLO mask + depth + cylinder fit (/perception/detect): position only, assumes the object stands upright
  auto      (default) pose if it is TRACKING, fresh, and agrees with the detector (pnp_core.object_grasp.pose_agrees:
            xy within pose_agree_m / pose_agree_frac of the object's size, yaw within pose_agree_deg); else detector
The grasp and the release are announced on /pick_place/grasp (std_msgs/String JSON, latched) so the pose node can
follow the object while it is in the hand (ATTACHED: pose = TCP pose x grasp offset) and look for it where it was
put down, instead of re-registering from scratch.

Close look (param wrist_refine, live; needs the wrist camera + pose_estimation:=true): the wrist camera (D405)
takes a close look before the grasp; the pose node refines the pose there (/perception/refine_pose) and the grasp is
re-planned from it. The view point: the pre-grasp pose (~13 cm) if the whole object fits the view there, else a
"look pose" backed off until it fills look_fill of the image (FoundationPose needs the whole object: the 28 cm
cracker box overfilled the close view and refined to 16-19 mm errors). Corrections larger than
max_refine_correction_m / _deg are rejected (the two cameras disagree: keep the first estimate).
After a FAILED attempt the next one always takes the close look (if there is a wrist camera), even with
wrist_refine off: retrying from the same estimate only repeats the same miss.

Missed grasps (up to max_grasp_attempts, then GRASP_FAILED): the jaws closed to the wrong width, the object slipped
out during the lift, or the straight-down APPROACH was blocked (the controller stopped the arm: a jaw came down ON
the object, a few mm off from the arm's own error or a stale pose). Each: back off straight up, home, look again.

Map mode (param map_mode, switchable at runtime; docs/learning/13_map_modes.md):
  continuous       obstacles update all the time (the camera -> Octomap, or the known-obstacle list)
  freeze_per_task  at the start of each task (arm at home, out of the camera's view) take a fresh look, then plan
                   the whole task against that snapshot: the moving arm and the carried object can't disturb the map.
                   A change monitor keeps comparing the live view with the snapshot; before each planned move
                   (pre-grasp, pre-place, home) change_policy decides: warn (log, keep the snapshot) | refresh (update
                   the snapshot, then plan) | stop (abort with SCENE_CHANGED, put the object back).

Calibration (param calibration_policy, live; docs/learning/16_calibration.md): the calibration monitor says whether the
fixed camera is still where its calibration put it (/calibration/status). At the start of a task, if it reports
DRIFT: warn (log it, run anyway) | stop (refuse with CALIBRATION_DRIFT: recalibrate, tools/calibrate.py fixed).

The grasp/place geometry comes from pnp_core (unit-tested). Robot I/O: robot_interface.py (standard ROS
interfaces only, identical on the real robot).
"""

import glob
import json
import math
import threading
import time
import traceback

import numpy as np
import rclpy
from pnp_interfaces.action import PickPlace
from pnp_interfaces.srv import RefinePose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from builtin_interfaces.msg import Duration
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from std_srvs.srv import SetBool
from rosbag2_interfaces.srv import Snapshot

from pnp_core.catalog import load_object, load_robot, load_yaml
from pnp_core.focus import focus_message
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix, wrap_angle
from pnp_core.grasp import Pose, grasp_yaws
from pnp_core.pickability import check_inflate, pick_check
from pnp_core import trials
from pnp_core.grasp_scoring import closing_side, joint_margin, score, tried_match, yaw_tolerance_deg
from pnp_core.object_grasp import (pose_agrees, ObjectGrasp, grasps_from_pose, nominal_grasp_offset, place_tcp_poses,
                                   place_tcp_yaws, shift_for_fixed_jaw,
                                   object_extents, rotation_error_deg, tcp_yaw_of, tilt_deg, upright_pose)
from pnp_core.reorient import current_rest, place_poses as turned_place_poses, rest_poses
from pnp_core.verification import GraspState, classify_grasp
from pnp_core.wrist import wrist_camera

from pnp_manipulation.robot_interface import RobotInterface, error_name

SUCCESS_CODE = 1          # moveit_msgs/MoveItErrorCodes.SUCCESS
GRASPED_ID = "grasped_object"
# The pick target as a MoveIt world object (its 6-D pose, else the detector's): MoveIt keeps its voxels out of the
# Octomap -- and only its: an obstacle beside it stays whole. The jaws may touch it only on the straight line down and
# at the grasp. After the release, the placed object likewise during the retreat. (Replaces the cut-out around each
# focus detection in pnp_core/cloud.py, which also erased the near side of a cube 1 cm beside a brick.)
TARGET_ID = "pick_target"
PLACED_ID = "placed_object"
OUTCOME_NAMES = {getattr(PickPlace.Result, k): k for k in dir(PickPlace.Result)
                 if k.isupper() and isinstance(getattr(PickPlace.Result, k), int)}


def self_robot_has_wrist() -> bool:
    """Is a wrist camera configured (config/robot_fr3.yaml wrist_camera)? -> the close look defaults to on."""
    return bool(load_robot().get("wrist_camera"))


class Abort(Exception):
    def __init__(self, outcome, message):
        super().__init__(message)
        self.outcome = outcome


class PickPlaceNode(Node):
    def __init__(self):
        super().__init__("pnp_pick_place")
        self.declare_parameter("scene", "config/scenes/basic.yaml")
        self.declare_parameter("max_grasp_attempts", 3)
        # score every grasp candidate (both hand directions: the fixed jaw on either side) by clearance, joint margin
        # at the grasp AND the place, yaw-error tolerance; a retry tries a different one (pnp_core/grasp_scoring.py).
        # Off: the first candidate that plans
        self.declare_parameter("grasp_scoring", False)
        # a staged final approach: stop closed_loop_stage_m above the grasp, look again with the wrist camera (a fresh
        # frame, the existing close-look refine), correct the last bit if the object is > closed_loop_correct_m off;
        # more than closed_loop_max_m / _deg off, or not found: back up and retry (UNCERTAIN after the last attempt).
        # Why: the servo errors make the arm sag differently 4 cm lower than where the close look was taken
        self.declare_parameter("closed_loop_approach", False)
        self.declare_parameter("closed_loop_stage_m", 0.02)
        self.declare_parameter("closed_loop_correct_m", 0.0015)
        self.declare_parameter("closed_loop_max_m", 0.006)
        self.declare_parameter("closed_loop_max_deg", 8.0)
        # the target object's shape grown by this on every side: its own voxels (1 cm, snapped to voxel centres) can
        # sit ~7 mm outside its real edge and blocked the open jaw at 3 cm from any obstacle; 8 mm keeps them inside.
        # An obstacle closer than that is the pickability check's job (raw points, no cut-out).
        self.declare_parameter("target_padding_m", 0.008)
        self.declare_parameter("target_clear_s", 0.5)         # after adding it: MoveIt clears its voxels (5 Hz cloud)
        self.declare_parameter("place_tolerance_m", 0.02)
        self.declare_parameter("use_known_obstacles", True)    # False: obstacles come from the Octomap (Layer B)
        self.declare_parameter("grasp_source", "auto")         # auto | pose | detector | ground_truth (sim only)
        self.declare_parameter("pose_wait_s", 2.5)             # DETECT waits this long for a fresh TRACKING pose (re-registering after the object jumped: ~1.7 s)
        # a pick with a hint (the workcell: "THIS one") takes only an object within this of the hint, from the detector
        # AND from the 6-D pose: the tracker follows an object wherever it went, so a moved one was picked anyway
        self.declare_parameter("pick_hint_radius_m", 0.06)
        self.declare_parameter("pose_agree_m", 0.015)          # auto: pose and detector must agree this well (xy)
        self.declare_parameter("pose_agree_frac", 0.35)        # ... and within this much of the object's small side
        self.declare_parameter("pose_agree_deg", 20.0)         # ... and in yaw (boxes)
        self.declare_parameter("max_grasp_tilt_deg", 20.0)     # object tilt still treated as standing / lying
        self.declare_parameter("max_place_tilt_deg", 15.0)     # VERIFY_PLACE (6-D pose): more tilt = it fell over
        self.declare_parameter("map_mode", "continuous")       # continuous | freeze_per_task (see the top of this file)
        self.declare_parameter("change_policy", "refresh")     # warn | refresh | stop  (freeze_per_task only)
        self.declare_parameter("map_settle_s", 1.0)            # fresh camera frames before a snapshot is taken
        self.declare_parameter("wrist_refine", bool(self_robot_has_wrist()))   # LOOK step (see the top of this file)
        self.declare_parameter("max_refine_correction_m", 0.03)
        self.declare_parameter("max_refine_correction_deg", 15.0)
        self.declare_parameter("look_fill", 0.6)               # object diameter / vertical field of view at the look pose
        self.declare_parameter("calibration_policy", "warn")   # warn | stop: what a DRIFT status does to a new task
        self.scene_cfg = load_yaml(self.get_parameter("scene").value)
        # the scene's defaults for the grasp strategy (grasping: {scoring, closed_loop}): the workcell page's room
        # check reads the same keys, so the page and the pick agree (ros2 param set still overrides for a test)
        gs = self.scene_cfg.get("grasping", {})
        self.set_parameters([Parameter("grasp_scoring", value=bool(gs.get("scoring", False))),
                             Parameter("closed_loop_approach", value=bool(gs.get("closed_loop", False)))])
        self.robot_cfg = load_robot()
        self.wrist_cfg = wrist_camera(self.robot_cfg)
        self.robot = RobotInterface(self, self.robot_cfg)
        # straight-line approach / lift heights [m]: the FR3's 10 / 15 cm would take a small arm out of its reach
        m = self.robot_cfg.get("grasp_motion", {})
        self._heights = {"approach_height": float(m.get("approach_height_m", 0.10)),
                         "lift_height": float(m.get("lift_height_m", 0.15))}
        self.busy = False
        self.target_in_scene = False
        self.planned_segments = {}   # what ("pick" / "place") -> [(name, arm joint waypoints)] of the last plan found
        self.holding = None          # GraspPlan of the object currently in the gripper (None = empty)
        self.held_from = None        # T_base_obj where the held object was picked (recovery puts it back there)
        self.last_pose, self.pose_state = None, None      # (T_base_obj, wall time received, pose-node state)
        cb = ReentrantCallbackGroup()
        # the status message carries the pose TOGETHER with the state it was computed in (a separate pose topic
        # could pair a stale pose with the previous tick's TRACKING)
        self.create_subscription(String, "/perception/pose_status", self._on_pose_status, 10, callback_group=cb)
        # everything standing on the table, no target cut-out (obstacle_cloud_node): the pickability check's input
        self.table_points = None                  # (wall time received, N x 3, base frame)
        # a PREVIEW's whole path (plan only): RViz "Planned motion" shows it (Show Trail / Loop Animation)
        self.pub_display = self.create_publisher(DisplayTrajectory, "/display_planned_path", 1)
        self.preview_message = ""
        self.create_subscription(PointCloud2, "/perception/table_points", self._on_table_points, 2, callback_group=cb)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_grasp = self.create_publisher(String, "/pick_place/grasp", latched)
        self.pub_focus = self.create_publisher(String, "/perception/focus", latched)
        self.current_obj = self.scene_cfg["target"]["object"]
        self.pub_focus.publish(String(data=self.current_obj))
        self.focus_changed_at = 0.0
        self.frozen = False          # this task runs against a frozen map
        self.changes = (None, 0.0)   # (latest /perception/map_changes report, wall time received)
        self.create_subscription(String, "/perception/map_changes",
                                 lambda m: setattr(self, "changes", (json.loads(m.data), time.time())), 10,
                                 callback_group=cb)
        self.freeze_cli = self.create_client(SetBool, "/perception/map_freeze", callback_group=cb)
        self.calibration = {}        # latest /calibration/status (the calibration monitor)
        self.create_subscription(String, "/calibration/status",
                                 lambda m: setattr(self, "calibration", json.loads(m.data)), latched, callback_group=cb)
        self.refine_cli = self.create_client(RefinePose, "/perception/refine_pose", callback_group=cb)
        # the failure ring buffer (bringup bag_buffer:=true, config/recording.yaml): saved when a task fails
        self.snapshot_cli = self.create_client(Snapshot, "/rosbag2_recorder/snapshot", callback_group=cb)
        self.task_id = None
        self.known = None            # latest known-obstacle layout (list of box dicts, pnp_core.obstacles)
        self.applied = (None, set())   # (layout json that is in MoveIt now, its collision object ids)
        self.scene_lock = threading.Lock()
        # Mutually exclusive group: layouts are applied one at a time, in the order they were published (with a
        # reentrant group, two threads could race and an OLDER layout could end up in MoveIt).
        self.create_subscription(String, "/scene/known_obstacles", self._on_known, 10,
                                 callback_group=MutuallyExclusiveCallbackGroup())
        ActionServer(self, PickPlace, "/pick_place", execute_callback=self._execute,
                     goal_callback=self._on_goal, cancel_callback=lambda _: CancelResponse.ACCEPT,
                     callback_group=ReentrantCallbackGroup())
        # The desk goes into MoveIt as soon as move_group answers, not only when the first task starts: until then
        # the depth camera's view of the desk itself filled the Octomap (a floor of voxels in RViz, seen on the SO-101).
        self._scene_timer = self.create_timer(2.0, self._initial_scene, callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info("pick & place action server ready on /pick_place")

    def _initial_scene(self):
        """Startup, once /apply_planning_scene is up: the desk as a collision object, then a fresh Octomap (voxels
        inserted before the desk was in MoveIt stay; its shape mask only filters new points)."""
        if self._scene_timer is None or not self.robot.scene.service_is_ready():
            return
        self._scene_timer.cancel()
        self._scene_timer = None
        try:
            t = self.scene_cfg["table"]
            self._setup_scene(t["center"][2] + t["size"][2] / 2)
            if not self.get_parameter("use_known_obstacles").value:
                self.robot.clear_octomap()
            self.get_logger().info("desk added to the planning scene")
        except Exception as e:  # noqa: BLE001  (a task adds it again anyway)
            self.get_logger().warn(f"adding the desk at startup failed ({e}); the first task adds it")

    # --- action plumbing ------------------------------------------------------------------------------------
    def _on_goal(self, _goal):
        return GoalResponse.REJECT if self.busy else GoalResponse.ACCEPT     # one task at a time

    def _execute(self, handle):
        self.busy = True
        t0 = time.time()
        result = PickPlace.Result(placement_error_m=float("nan"))
        self.task_id = time.strftime("%Y%m%d-%H%M%S-") + f"{int(t0 * 1000) % 1000:03d}"
        self.trial = {"s_detect": 0.0, "s_plan": 0.0, "s_look": 0.0, "jaw_width": None, "expected_width": None,
                      "chosen": None, "corrections": [], "candidates": []}
        self.tried_yaws, self.grasp_clearance = [], None
        # a hand in the workspace pauses the arm (RobotInterface): show it, log it, and let a cancel through
        self.robot.pause_listener = lambda state, info: self._on_pause(handle, state, info)
        self.robot.pause_poll = lambda: self._fb_cancel(handle)
        try:
            self._run(handle, result)
            result.outcome = PickPlace.Result.SUCCESS
            result.message = (self.preview_message if handle.request.preview else
                              "grabbed and lifted" if handle.request.grab_only else "placed at goal")
            handle.succeed()
        except Abort as a:
            result.outcome, result.message = a.outcome, str(a)
            self._unfreeze_map(settle=True)          # recovery plans with the CURRENT scene
            self._safe_recover()
            handle.canceled() if a.outcome == PickPlace.Result.CANCELED else handle.abort()
        except Exception as e:  # noqa: BLE001
            result.outcome, result.message = PickPlace.Result.ERROR, f"{type(e).__name__}: {e}"
            self.get_logger().error(traceback.format_exc())
            self._unfreeze_map(settle=True)
            self._safe_recover()
            handle.abort()
        finally:
            self._unfreeze_map()
            self.robot.pause_listener = self.robot.pause_poll = None
            self.busy = False
        result.outcome_name = OUTCOME_NAMES.get(result.outcome, "?")
        result.duration_s = float(time.time() - t0)
        if not handle.request.preview:
            self._log_trial(handle.request, result)
        self.get_logger().info(f"pick_place: {result.outcome_name} ({result.message}) in {result.duration_s:.1f} s, "
                               f"placement error {result.placement_error_m * 1000:.1f} mm")
        return result

    def _grip(self, width, expected, grasp_cfg):
        """Holding? The jaw reading (minus the robot's gripper.width_offset_m: a real servo's reading bias) vs the
        object's width, within its grasp.width_tol_m (bricks 3 mm: 26 mm on a 15.8 mm brick is a corner or the
        studs, not a grip; default 10 mm)."""
        off = float(self.robot_cfg["gripper"].get("width_offset_m", 0.0))
        return classify_grasp(width - off, expected, tol=float(grasp_cfg.get("width_tol_m", 0.01)))

    def _score_grasps(self, handle, attempt, grasps, T_obj, obj, grasp_cfg, place_candidates, grab_only, home):
        """grasp_scoring: every candidate scored (pnp_core/grasp_scoring.py), best first; the table goes to the trial
        log. Rejected (no IK at the grasp / the place) are dropped unless all are."""
        R, cfg = self.robot, self.robot_cfg
        joints, limits = cfg["arm"]["joints"], cfg["arm"].get("position_limits", {})
        clear = self.grasp_clearance if self.grasp_clearance and len(self.grasp_clearance) == len(grasps) \
            else [None] * len(grasps)
        ext = object_extents(obj)
        fixed = R.gripper_model.approach_clearance
        room = max(0.0, min(fixed, self._open_width(grasp_cfg) - float(grasp_cfg["width_m"]) - fixed))
        rows = []
        narrowest = min(g.width_m for g in grasps)
        for i, g in enumerate(grasps):
            yaw = tcp_yaw_of(g.T_base_tcp[:3, :3])
            row = {"idx": i, "_g": g, "closing_side": closing_side(g.T_base_tcp, T_obj),
                   "yaw": round(math.degrees(yaw), 1), "clearance": clear[i], "tried": tried_match(yaw, self.tried_yaws),
                   "chosen": 0, "reject": None, "narrow": narrowest / max(g.width_m, 1e-6)}
            pl = g.plan(cfg["gripper"]["max_width_m"], grasp_cfg["force_n"], **self._heights)
            q = R.ik(pl.grasp.position, pl.grasp.quat_xyzw, seed=home)
            if q is None:
                row["reject"] = "no IK at the grasp"
            else:
                row["roll_pick"] = float(q[-1])
                m = joint_margin(q, joints, limits)
                if not grab_only:
                    pc = place_candidates(g.T_tcp_obj, g.width_m)
                    qp = R.ik(pc[0][1].position, pc[0][1].quat_xyzw, seed=home) if pc else None
                    if qp is None:
                        row["reject"] = "no IK at the place"
                    else:
                        row["roll_place"] = float(qp[-1])
                        m = min(m, joint_margin(qp, joints, limits))
                row["joint_margin"] = m
            length = float(ext[0] + ext[1] - g.width_m) if g.kind == "box" else g.width_m
            row["yaw_tol"] = yaw_tolerance_deg(g.width_m, length, room)
            row["score"] = None if row["reject"] else round(score(row), 3)
            rows.append(row)
        # a grip much wider than needed (a 2x4 across its length) only if no narrow one is possible: as a RETRY it
        # tipped the brick over (demo, 03:27 and 03:31)
        if any(not r["reject"] and r["narrow"] >= 0.75 for r in rows):
            for r in rows:
                if not r["reject"] and r["narrow"] < 0.75:
                    r["reject"], r["score"] = "wider grip than needed", None
        self.trial["candidates"] = rows
        ok = sorted((r for r in rows if not r["reject"]), key=lambda r: -r["score"])
        if not ok:
            self._fb(handle, "DETECT", attempt, f"grasp scoring: every candidate rejected "
                                                f"({'; '.join(r['reject'] for r in rows)}): planning decides")
            return grasps
        b = ok[0]
        self._fb(handle, "DETECT", attempt,
                 f"grasp scoring: {len(ok)} of {len(rows)} candidates, best {b['closing_side']} yaw {b['yaw']:.0f} deg "
                 f"score {b['score']:.2f} (clearance {(b['clearance'] or 0) * 1000:.0f} mm, joint margin "
                 f"{b['joint_margin']:.2f} rad, yaw tolerance {b['yaw_tol']:.1f} deg{', retry: untried' if self.tried_yaws else ''})")
        return [r["_g"] for r in ok]

    def _log_trial(self, goal, result):
        """This task -> the trial log (pnp_core/trials.py, /data/trials.sqlite). Never fails the task."""
        try:
            tr, c = self.trial, trials.connect()
            run = trials.current_run(c, self.scene_cfg.get("name", "?"))
            total = float(result.duration_s)
            move = max(0.0, total - tr["s_detect"] - tr["s_plan"] - tr["s_look"])
            a = trials.add_attempt(
                c, run, object=self.current_obj, result=result.outcome_name,
                failure=trials.classify_failure(result.outcome_name, result.message), message=result.message[:300],
                jaw_width=tr["jaw_width"], expected_width=tr["expected_width"], chosen=tr["chosen"],
                retries=max(0, int(result.grasp_attempts) - 1), corrections=tr["corrections"],
                s_detect=tr["s_detect"], s_plan=tr["s_plan"], s_move=move, s_look=tr["s_look"], s_total=total,
                task=self.task_id)
            if tr["candidates"]:
                trials.add_candidates(c, run, a, [{**r, "yaw_tol": r.get("yaw_tol"), "_g": None}
                                                  for r in tr["candidates"]])
            c.close()
            # something MOVED and it went wrong: keep the last seconds of every sensor (ring buffer, if running)
            if result.outcome_name not in ("SUCCESS", "CANCELED", "BLOCKED"):
                self._snapshot(run, a)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"trial log: {e}")

    def _snapshot(self, run, attempt):
        """Write the failure ring buffer to its bag and link it to this attempt (artifacts). Never fails the task."""
        if not self.snapshot_cli.service_is_ready():
            return
        try:
            res = self.robot.wait(self.snapshot_cli.call_async(Snapshot.Request()), 15.0)
            if res is None or not res.success:
                self.get_logger().warn("ring buffer: snapshot failed")
                return
            bags = sorted(glob.glob("/data/bags/ring_*"))     # ring_<YYYYmmdd_HHMMSS>: the newest bringup's last
            trials.artifact("ring_snapshot", bags[-1] if bags else "/data/bags", run=run, attempt=attempt,
                            task=self.task_id, note=f"snapshot at {time.strftime('%H:%M:%S')} (the last seconds "
                                                     f"before the failure; ros2 bag play --start-offset to jump)")
            self.get_logger().info(f"ring buffer: snapshot saved for attempt {attempt}")
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"ring buffer: {e}")

    def _event(self, kind, severity="info", **data):
        """An event of this task -> the trial log (pnp_core/trials.py events). Never fails the task."""
        trials.event("pick_place", kind, data, severity, task=self.task_id)

    def _on_pause(self, handle, state, info):
        """RobotInterface stopped (state "paused") / continues ("resumed") a move for a hand in the workspace."""
        detail = (f"a hand in the workspace: stopped, waiting until it's clear ({info.get('why', '')})"
                  if state == "paused" else f"the workspace is clear: continuing (paused {info.get('waited_s', 0)} s)")
        handle.publish_feedback(PickPlace.Feedback(state=state.upper(), attempt=0, detail=detail))
        self.get_logger().info(f"[{state.upper()}] {detail}")
        self._event(f"hand_{state}", "warn" if state == "paused" else "info", **info)

    @staticmethod
    def _fb_cancel(handle):
        if handle.is_cancel_requested:
            raise Abort(PickPlace.Result.CANCELED, "canceled (while paused for a hand)")

    def _fb(self, handle, state, attempt=0, detail=""):
        if handle.is_cancel_requested:
            raise Abort(PickPlace.Result.CANCELED, "canceled")
        handle.publish_feedback(PickPlace.Feedback(state=state, attempt=attempt, detail=detail))
        self.get_logger().info(f"[{state}] {detail}")

    def _move(self, outcome, what, fn, *args, **kw):
        code = fn(*args, **kw)
        if error_name(code) == "START_STATE_IN_COLLISION" and not self.get_parameter("use_known_obstacles").value:
            # Layer B: the robot IS where it is, so a "collision" of its current state with PERCEIVED obstacles is
            # a perception artefact (e.g. a stale voxel inside the can it just grasped). Clear the Octomap, let the
            # camera refill it, try once more.
            self.get_logger().warn(f"{what}: start state collides with the Octomap -> clearing it and retrying")
            self.robot.clear_octomap()
            if self.frozen:                   # the snapshot is gone with it: take a new one from the live camera
                self._refresh_map()
            code = fn(*args, **kw)
        if code != SUCCESS_CODE:
            extra = ""
            if error_name(code) == "CONTROL_FAILED":   # the controller couldn't follow: where did the arm get stuck?
                extra = f" (arm stopped at q={np.round(self.robot.arm_q(), 3).tolist()}"
                extra += f", target q={np.round(args[0], 3).tolist()})" if fn == self.robot.move_joints else ")"
            raise Abort(outcome, f"{what} failed: {error_name(code)}{extra}")

    # --- the task -------------------------------------------------------------------------------------------
    def _run(self, handle, result):
        goal = handle.request
        R, cfg = self.robot, self.robot_cfg
        # WHICH object: the class asked for (several objects on the table: the workcell page) -> its catalog entry;
        # default the scene's target. Perception follows it (/perception/focus: the detector's mask, the 6-D pose
        # node's mesh, the obstacle map's cut-out: the other objects stay obstacles).
        obj_name = self._object_for(goal.object_class)
        obj = load_object(obj_name)
        label = obj["yolo_class"]
        grasp_cfg = obj["grasp"]
        if obj_name != self.current_obj:      # the pose node swaps meshes and registers afresh (~3 s): wait for it
            self.focus_changed_at = time.time()
        self.current_obj = obj_name
        hint = (goal.pick_x, goal.pick_y) if goal.use_pick_hint else None
        self.pub_focus.publish(String(data=focus_message(obj_name, hint)))    # pnp_core/focus.py: + which instance
        home = cfg["arm"]["home"]
        prof = cfg["arm"]["speed_profiles"].get(goal.speed_profile or "dev")
        if prof is None:
            raise Abort(PickPlace.Result.ERROR, f"unknown speed_profile {goal.speed_profile!r}")
        R.free_scaling, lin = prof["free"], prof["linear"]          # free-space moves / straight-line moves
        if goal.use_default_place:
            place_xy, place_yaw = self.scene_cfg["goal"]["xy"], math.radians(self.scene_cfg["goal"]["yaw_deg"])
        else:
            place_xy, place_yaw = [goal.place_x, goal.place_y], goal.place_yaw
        table_z = self.scene_cfg["table"]["center"][2] + self.scene_cfg["table"]["size"][2] / 2
        rest = None                                            # set: put it down on this face (turn it over)
        if goal.place_rest:
            rests = rest_poses(obj)
            if goal.place_rest not in rests:
                raise Abort(PickPlace.Result.ERROR, f"'{obj_name}' has no resting face '{goal.place_rest}' "
                                                    f"({', '.join(rests)})")
            rest = rests[goal.place_rest]

        if not R.wait_ready():
            raise Abort(PickPlace.Result.ERROR, "robot not ready (no joint states / move_group)")
        self._check_calibration()
        self._setup_scene(table_z)
        self._sync_obstacles()                                 # in case the newest layout is still being applied
        R.detach(GRASPED_ID)                                   # clean state if a previous run aborted mid-way
        self._scene_objects_off()
        if self.pose_state == "ATTACHED":                      # ... and the pose node must not think we hold it
            self._grasp_event("lost")

        self._fb(handle, "HOME", detail="start from the observation pose (arm out of the camera's way)")
        if np.abs(R.arm_q() - np.array(home)).max() > 0.02 and not goal.preview:     # a preview never moves
            self._move(PickPlace.Result.MOTION_FAILED, "move home", R.move_joints, home)
        self._freeze_map(handle)                               # map_mode freeze_per_task: snapshot now

        # Check the GOAL is reachable before touching the object (plan only, nothing moves): otherwise we'd
        # find out while holding it and have to put it back.
        sym = grasp_cfg.get("symmetric_yaw", False)
        clearance = obj.get("place_clearance_m", 0.005)

        def place_candidates(T_tcp_obj, width=None):
            """Gripper poses that set the held object (grasp offset T_tcp_obj) down at the goal, best yaw first.
            With a face to turn it to (rest): the hand turned with it (pnp_core/reorient.py), e.g. horizontal."""
            if rest is not None:
                per = rest.period_deg
                yaws = ([place_yaw + 2 * math.pi * k / 8 for k in range(8)] if per is None else
                        [place_yaw + math.radians(per) * k for k in range(int(round(360 / per)))])
                out = []
                for _, T_tcp, T_o in turned_place_poses(T_tcp_obj, obj, rest, place_xy, table_z, clearance, yaws,
                                                        cfg["gripper"], width or float(grasp_cfg.get("width_m", 0.0))):
                    q = matrix_to_quat(T_tcp[:3, :3])
                    pl = Pose(T_tcp[:3, 3].copy(), q)
                    # a turned (horizontal) hand starts higher: the forearm comes low over the desk otherwise
                    pre = Pose(T_tcp[:3, 3] + [0.0, 0.0, self._heights["approach_height"] + 0.05], q)
                    out.append((pre, pl, (pre, pl, T_o)))
                return out
            yaws = place_tcp_yaws(T_tcp_obj, obj, place_yaw, self._yaws(place_yaw, sym), R.tcp_yaw())
            return [(pre_, place_, (pre_, place_, T_obj_))
                    for _, pre_, place_, T_obj_ in place_tcp_poses(T_tcp_obj, obj, place_xy, table_z, clearance, yaws,
                                                                   approach_height=self._heights["approach_height"])]
        # before we have it: assume the standard grasp of the upright object
        nominal = shift_for_fixed_jaw([ObjectGrasp("nominal", np.eye(4), nominal_grasp_offset(obj),
                                                   float(obj.get("grasp", {}).get("width_m", 0.0)))],
                                      R.gripper_model.held_offset_y,
                                      R.gripper_model.approach_clearance)[0].T_tcp_obj
        if rest is None and not goal.grab_only:   # turning it over: checked once we see it (its grasp decides)
            self._plan_approach(place_candidates(nominal), lin, f"goal ({place_xy[0]:.3f}, {place_xy[1]:.3f})")

        attempts, max_attempts = 0, self.get_parameter("max_grasp_attempts").value
        force_look = False                 # after a failed attempt: take the wrist camera's close look regardless
        hint = (goal.pick_x, goal.pick_y)  # retries: where the object was last seen (a missed grasp can nudge it)
        while True:
            attempts += 1
            result.grasp_attempts = attempts
            # DETECT -----------------------------------------------------------------------------------------
            self._fb(handle, "DETECT", attempts, f"looking for '{label}'")
            t_req = time.time()
            det, hint_r = None, self.get_parameter("pick_hint_radius_m").value
            for _ in range(0 if self.get_parameter("grasp_source").value == "ground_truth" else 3):
                found = R.detect(label)
                if found and goal.use_pick_hint:                 # the one the user chose: nearest the hint
                    near = [d for d in found if math.hypot(d[0][0] - hint[0], d[0][1] - hint[1]) < hint_r]
                    found = sorted(near, key=lambda d: math.hypot(d[0][0] - hint[0], d[0][1] - hint[1]))[:1]
                if found:
                    det = max(found, key=lambda d: d[1])         # highest confidence
                    break
                time.sleep(0.5)
            T_obj, source = self._object_pose(det, t_req, label, obj)
            center = T_obj[:3, 3]
            if goal.use_pick_hint and math.hypot(center[0] - hint[0], center[1] - hint[1]) > hint_r:
                raise Abort(PickPlace.Result.NO_OBJECT,
                            f"no '{label}' within {hint_r * 100:.0f} cm of the chosen spot ({hint[0]:.3f}, "
                            f"{hint[1]:.3f}); the nearest is at ({center[0]:.3f}, {center[1]:.3f}): moved? scan again")
            hint = (float(center[0]), float(center[1]))
            grasps, why = grasps_from_pose(T_obj, obj, self._yaws(0.0, sym), R.tcp_yaw(),
                                           self.get_parameter("max_grasp_tilt_deg").value, cfg["gripper"]["max_width_m"])
            grasps = shift_for_fixed_jaw(grasps, R.gripper_model.held_offset_y,
                                      R.gripper_model.approach_clearance)      # one moving jaw: see there
            if not grasps:
                raise Abort(PickPlace.Result.UNREACHABLE, f"no grasp for the object as it lies ({source}): {why}")
            # PICKABLE? the open hand's space at each grasp vs what stands on the table -- before anything moves
            self.grasp_clearance = None
            grasps = self._pickable_grasps(handle, attempts, grasps, T_obj, obj, sym, t_req)
            self.trial["s_detect"] += time.time() - t_req
            if rest is None and self.get_parameter("grasp_scoring").value:
                grasps = self._score_grasps(handle, attempts, grasps, T_obj, obj, grasp_cfg, place_candidates,
                                            goal.grab_only, home)
            if rest is not None:
                # only grasps whose turned hand can set it down on that face (geometry: hand not pointing up, nothing
                # below the table), and the first of those whose placement the ARM reaches (planned, nothing moves)
                now = current_rest(T_obj[:3, :3], obj) or "leaning"
                turnable = [g for g in grasps if place_candidates(g.T_tcp_obj, g.width_m)]
                if not turnable:
                    raise Abort(PickPlace.Result.UNREACHABLE,
                                f"no grasp of the object as it lies ({now}) can set it down {rest.label} in one move")
                for i, g in enumerate(turnable):
                    try:
                        self._plan_approach(place_candidates(g.T_tcp_obj, g.width_m), lin,
                                            f"goal {rest.label} ({place_xy[0]:.3f}, {place_xy[1]:.3f})", tries=3,
                                            straight=False)
                        grasps = [g] + [x for x in turnable if x is not g]
                        break
                    except Abort:
                        if i == len(turnable) - 1:
                            raise
                why += f"; turning it from {now} to {rest.label}"
            self._fb(handle, "DETECT", attempts, f"grasp from {source}: {why}")
            result.picked_at = [float(center[0]), float(center[1]), float(grasps[0].T_base_tcp[2, 3]
                                                                         + grasp_cfg["depth_below_top_m"])]
            self._target_on(T_obj, obj)
            plans = [(g, g.plan(cfg["gripper"]["max_width_m"], grasp_cfg["force_n"], **self._heights)) for g in grasps]

            # APPROACH ---------------------------------------------------------------------------------------
            self._check_scene(handle, "MOVE_PREGRASP", attempts)
            R.gripper(self._open_width(grasp_cfg), grasp_cfg["force_n"])
            self._fb(handle, "MOVE_PREGRASP", attempts, f"above object at ({center[0]:.3f}, {center[1]:.3f})")
            try:
                q_pre, (g, plan) = self._plan_approach([(pl.pregrasp, pl.grasp, (g, pl)) for g, pl in plans], lin,
                                                       "pick")
            except Abort as e:
                contacts = getattr(R, "_last_contacts", "")
                if "<octomap>" in contacts:    # no way down that misses what the camera sees beside it: blocked
                    raise Abort(PickPlace.Result.BLOCKED, f"no collision-free way down to the object: an obstacle "
                                                          f"beside it is in the way ({contacts}); {e}")
                raise
            self.trial["chosen"] = next((i for i, (gg, _) in enumerate(plans) if gg is g), None)
            for row in self.trial["candidates"]:
                if row.get("_g") is not None:
                    row["chosen"] = int(row["_g"] is g)
            if goal.preview:                              # plan only: show the whole path, report, move nothing
                self._preview(handle, g, place_candidates, lin, rest)
                return
            self._move(PickPlace.Result.UNREACHABLE, "move to pre-grasp", R.move_joints, q_pre)
            validated = (q_pre, g, plan, T_obj)          # this approach's straight line was checked before moving
            g, plan, T_obj = self._close_look(handle, attempts, g, plan, T_obj, obj, grasp_cfg, lin, force_look)
            self._target_on(T_obj, obj)                    # where the close look put it
            result.picked_at = [float(T_obj[0, 3]), float(T_obj[1, 3]),
                                float(plan.grasp.position[2] + grasp_cfg["depth_below_top_m"])]
            self._fb(handle, "APPROACH", attempts, "straight down")
            self._target_contact(True)                     # the open jaws reach around it: only from here to the grasp
            if self.get_parameter("closed_loop_approach").value and self.wrist_cfg is not None:
                staged = self._staged_descent(handle, attempts, g, plan, T_obj, obj, grasp_cfg, lin)
                if staged is None:                         # too far off / lost: backed up; try again from the top
                    self._target_contact(False)
                    R.move_joints(home)
                    self.tried_yaws.append(tcp_yaw_of(g.T_base_tcp[:3, :3]))
                    if attempts >= max_attempts:
                        raise Abort(PickPlace.Result.UNCERTAIN, f"the staged approach couldn't confirm the object "
                                                                f"under the jaws, after {attempts} attempts")
                    force_look = True
                    continue
                g, plan, T_obj = staged
            try:
                try:
                    self._move(PickPlace.Result.MOTION_FAILED, "approach", R.move_pose,
                               plan.grasp.position, plan.grasp.quat_xyzw, linear=True, scaling=lin)
                except Abort as e:
                    if "CONTROL_FAILED" in str(e) or plan is validated[2]:
                        raise
                    # the close look's CORRECTED approach couldn't be planned (a 7-joint arm re-planned into another
                    # arm configuration whose straight line has no IK, seen on the FR3): the validated one can
                    self._fb(handle, "APPROACH", attempts, f"corrected approach not plannable ({e}): back to the "
                                                           f"first one")
                    q_pre, g, plan, T_obj = validated
                    self._move(PickPlace.Result.UNREACHABLE, "back to pre-grasp", R.move_joints, q_pre)
                    self._move(PickPlace.Result.MOTION_FAILED, "approach", R.move_pose,
                               plan.grasp.position, plan.grasp.quat_xyzw, linear=True, scaling=lin)
            except Abort as e:
                if "CONTROL_FAILED" not in str(e):           # planning / collision problems stay hard failures
                    raise
                # the arm was STOPPED on the way down: a jaw landed on the object -> a missed grasp, not a fault
                above = float(R.tcp_pose()[0][2] - plan.grasp.position[2])
                detail = f"approach blocked {above * 1000:.0f} mm above the grasp (a jaw landed on the object?)"
                self._fb(handle, "VERIFY_GRASP", attempts, detail)
                self.get_logger().warn(f"attempt {attempts}: {detail}")
                R.move_pose(plan.pregrasp.position, plan.pregrasp.quat_xyzw, linear=True, scaling=lin)
                self._target_contact(False)
                R.move_joints(home)
                self.tried_yaws.append(tcp_yaw_of(g.T_base_tcp[:3, :3]))
                if attempts >= max_attempts:
                    raise Abort(PickPlace.Result.GRASP_FAILED, f"{detail}, after {attempts} attempts")
                force_look = True
                continue

            # GRASP + VERIFY ---------------------------------------------------------------------------------
            self._fb(handle, "GRASP", attempts, f"closing with {grasp_cfg['force_n']:.0f} N")
            width = R.close_on_object(grasp_cfg["force_n"])      # returns once the fingers stopped moving
            state = self._grip(width, plan.expected_width_m, grasp_cfg)
            self.trial.update(jaw_width=float(width), expected_width=float(plan.expected_width_m))
            self._fb(handle, "VERIFY_GRASP", attempts, f"finger width {width * 1000:.1f} mm -> {state.value}")
            self._event("grip", "info" if state is GraspState.HOLDING else "warn", attempt=attempts, phase="close",
                        object=self.current_obj, width_mm=round(width * 1000, 2),
                        expected_mm=round(plan.expected_width_m * 1000, 2),
                        tol_mm=round(float(grasp_cfg.get("width_tol_m", 0.01)) * 1000, 2), state=state.value)
            if state is GraspState.HOLDING:
                # MoveIt: the object now moves with the hand (cylinder axis = the object's up axis)
                self._target_off()
                R.attach_object(GRASPED_ID, obj, g.T_tcp_obj)
                self.holding, self.held_from = plan, T_obj       # recovery puts it back here if anything fails
                self._grasp_event("attached", T_tcp_obj=g.T_tcp_obj)
                self._fb(handle, "LIFT", attempts)
                self._move(PickPlace.Result.MOTION_FAILED, "lift", R.move_pose,
                           plan.lift.position, plan.lift.quat_xyzw, linear=True, scaling=lin)
                if self._grip(R.gripper_width(), plan.expected_width_m, grasp_cfg) is GraspState.HOLDING:
                    break                                        # we have it
                R.detach(GRASPED_ID)
                self.holding = None
                self._grasp_event("lost")                        # it fell somewhere: the pose node looks again
                detail = "object slipped out during the lift"
                self._event("grip", "warn", attempt=attempts, phase="lift", object=self.current_obj,
                            width_mm=round(R.gripper_width() * 1000, 2), state="slipped")
            else:
                detail = f"grasp {state.value} (width {width * 1000:.1f} mm)"
            # failed attempt: open, back off straight up, go home, try again
            self.get_logger().warn(f"attempt {attempts}: {detail}")
            R.gripper(self._open_width(grasp_cfg), grasp_cfg["force_n"])
            R.move_pose(plan.pregrasp.position, plan.pregrasp.quat_xyzw, linear=True, scaling=lin)
            self._target_contact(False)
            R.move_joints(home)
            self.tried_yaws.append(tcp_yaw_of(g.T_base_tcp[:3, :3]))
            if attempts >= max_attempts:
                raise Abort(PickPlace.Result.GRASP_FAILED, f"{detail}, after {attempts} attempts")
            force_look = True

        if goal.grab_only:                 # the grab soak: it held through the lift. Put it down here, go home
            self._fb(handle, "RELEASE", attempts, "grab only: holding after the lift -> released where it is")
            R.gripper(self._open_width(grasp_cfg), grasp_cfg["force_n"])
            R.detach(GRASPED_ID)
            self.holding = None
            self._grasp_event("lost")
            R.move_joints(home)
            return
        # TRANSPORT + PLACE -----------------------------------------------------------------------------------
        self._fb(handle, "MOVE_PREPLACE", attempts, f"above goal at ({place_xy[0]:.3f}, {place_xy[1]:.3f})")
        # the MEASURED grasp offset: where the object really sits in the hand decides where the hand goes
        self._check_scene(handle, "MOVE_PREPLACE", attempts)
        free_before = R.free_scaling
        if rest is not None:               # turning it over: slowly (it can swing round the fingers)
            R.free_scaling = min(R.free_scaling, float(cfg["arm"].get("turned_carry_scaling", R.free_scaling)))
        q_pre, (pre, place, T_obj_placed) = self._plan_approach(place_candidates(g.T_tcp_obj, g.width_m), lin, "place",
                                                                tries=2 if rest is None else 4, straight=rest is None)
        if rest is None or R.move_joints(q_pre, ptp=True) != SUCCESS_CODE:     # turned: a short joint-space line
            self._move(PickPlace.Result.UNREACHABLE, "move to pre-place", R.move_joints, q_pre)   # first if it can
        if self._grip(R.gripper_width(), plan.expected_width_m, grasp_cfg) is not GraspState.HOLDING:
            R.detach(GRASPED_ID)
            self.holding = None
            self._grasp_event("lost")
            raise Abort(PickPlace.Result.DROPPED, f"object lost during transport (width {R.gripper_width() * 1000:.1f} mm)")
        self._fb(handle, "DESCEND", attempts)
        if rest is None:
            self._move(PickPlace.Result.MOTION_FAILED, "descend", R.move_pose, place.position, place.quat_xyzw,
                       linear=True, scaling=lin)
        else:   # turned: a joint-space line to the joints the plan found (no IK again: it can miss them), slowly
            self._move(PickPlace.Result.MOTION_FAILED, "descend", R.move_joints, self.last_q_target, ptp=True)
        self._fb(handle, "RELEASE", attempts)
        R.free_scaling = free_before
        R.gripper(self._open_width(grasp_cfg), grasp_cfg["force_n"])
        R.detach(GRASPED_ID)
        self.holding = None
        settled = T_obj_placed.copy()
        settled[2, 3] -= clearance                              # released `clearance` above the table: it settles
        self._grasp_event("released", T_base_obj=settled)
        self.robot.add_object(PLACED_ID, obj, settled, self.get_parameter("target_padding_m").value)
        self.robot.allow_contact(PLACED_ID, self._hand(), True)      # the jaws leave around it
        self._fb(handle, "RETREAT", attempts)
        if rest is None:
            self._move(PickPlace.Result.MOTION_FAILED, "retreat", R.move_pose, pre.position, pre.quat_xyzw,
                       linear=True, scaling=lin)
        else:
            self._retreat_turned(handle, attempts, place, lin)
        self._scene_objects_off()                       # from now on it's seen like anything else on the table
        self._check_scene(handle, "HOME", attempts)
        self._fb(handle, "HOME", attempts)
        self._move(PickPlace.Result.MOTION_FAILED, "move home", R.move_joints, home)

        # VERIFY PLACE --------------------------------------------------------------------------------------------
        self._fb(handle, "VERIFY_PLACE", attempts, "measuring the object at the goal")
        p, how = self._placed_position(label, place_xy, obj, rest)
        err = float(np.hypot(p[0] - place_xy[0], p[1] - place_xy[1]))
        result.placed_at = [float(v) for v in p]
        result.placement_error_m = err
        self._fb(handle, "VERIFY_PLACE", attempts, f"{err * 1000:.1f} mm from the goal ({how})")
        if err > self.get_parameter("place_tolerance_m").value:
            raise Abort(PickPlace.Result.PLACED_OFF_TARGET, f"placed {err * 1000:.1f} mm from the goal ({how})")

    # --- object pose: 6-D pose node first, detector as the fallback ------------------------------------------------
    def _on_pose_status(self, m: String):
        d = json.loads(m.data)
        self.pose_state = d.get("state")
        if d.get("object") not in (None, self.current_obj):   # still the previous object (switching meshes)
            self.pose_state = f"estimating '{d.get('object')}', not '{self.current_obj}'"
            return
        if "T_base_obj" in d:
            self.last_pose = (np.array(d["T_base_obj"]), time.time(), self.pose_state)

    def _fresh_pose(self, since, wait_s):
        """-> (T_base_obj | None, why): a pose received after `since` while the pose node was TRACKING."""
        end = time.time() + wait_s
        while True:
            lp = self.last_pose
            if lp is not None and lp[1] >= since and lp[2] == "TRACKING":
                return lp[0], "ok"
            if time.time() >= end:
                return None, ("no 6-D pose received (pose_estimation:=true?)" if lp is None
                              else f"no fresh TRACKING pose (pose node: {self.pose_state})")
            time.sleep(0.05)

    def _object_for(self, object_class: str) -> str:
        """The catalog name of the object to pick: the scene object (target or extra_objects) of that class, else any
        catalog object of that class, else the scene's target (empty class)."""
        target = self.scene_cfg["target"]["object"]
        if not object_class:
            return target
        catalog = load_yaml("config/objects.yaml")
        scene = [target] + [e["object"] for e in self.scene_cfg.get("extra_objects", [])]
        for name in scene + list(catalog):
            if catalog.get(name, {}).get("yolo_class") == object_class:
                return name
        raise Abort(PickPlace.Result.NO_OBJECT, f"no object of class '{object_class}' in config/objects.yaml")

    def _object_pose(self, det, since, label, obj):
        """The object pose to grasp from -> (T_base_obj, description of the source). Raises Abort(NO_OBJECT)."""
        src, why = self.get_parameter("grasp_source").value, "grasp_source=detector"
        self.disputed = None              # set: the 6-D pose and the detector disagreed (-> the close look must settle it)
        if src == "ground_truth":
            return self._ground_truth_pose(), "ground truth (sim)"
        if src != "detector":
            wait = self.get_parameter("pose_wait_s").value
            wait = max(wait, 6.0 - (time.time() - self.focus_changed_at))     # just switched objects: registering
            T, why = self._fresh_pose(since, wait)
            if T is not None and det is not None and src == "auto":    # tracking slid off / stale / wrong object?
                ok, msg = pose_agrees(T, det[0][:2], det[2], obj, self.get_parameter("pose_agree_m").value,
                                      self.get_parameter("pose_agree_frac").value,
                                      self.get_parameter("pose_agree_deg").value)
                if not ok:
                    T, why, self.disputed = None, msg, msg
            if T is not None:
                return T, "6-D pose"
            if src == "pose":
                raise Abort(PickPlace.Result.NO_OBJECT, f"no usable 6-D pose of '{label}': {why}")
        if det is None:
            raise Abort(PickPlace.Result.NO_OBJECT,
                        f"no '{label}' detected" + ("" if src == "detector" else f" (6-D pose: {why})"))
        c = det[0]                         # the detector assumes the object stands upright; a box's yaw = long side
        t = self.scene_cfg["table"]
        height = 2 * (c[2] - (t["center"][2] + t["size"][2] / 2))
        if abs(height - obj["dims_m"][2]) > 0.02:
            # it rests on another face (turned over): an upright grasp from the detector would be centimeters off
            raise Abort(PickPlace.Result.UNCERTAIN, f"the detector sees it on another face ({height * 1000:.0f} mm "
                                                    f"tall) and there's no 6-D pose ({why}): not grasping blind")
        return upright_pose(obj, c[:2], c[2], yaw=det[2]), f"detector ({why})"

    def _ground_truth_pose(self):
        """grasp_source=ground_truth (SIM ONLY): the object's true pose from Isaac (TF base -> 'target'). Tests the
        manipulation side without perception: if picks fail like this, it's not the camera's fault."""
        try:
            frame = next((e["name"] for e in self.scene_cfg.get("extra_objects", []) if e["object"] == self.current_obj
                          and self.current_obj != self.scene_cfg["target"]["object"]), "target")   # sim TF: its name
            t = self.robot.tf.lookup_transform(self.robot_cfg["frames"]["base"], frame, Time()).transform
        except Exception as e:  # noqa: BLE001
            raise Abort(PickPlace.Result.NO_OBJECT, f"no ground-truth TF of the target (sim only): {e}")
        return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                              [t.translation.x, t.translation.y, t.translation.z])

    def _face_check(self, T, obj, rest):
        """Placed on a face other than the one asked for (turning it over: `rest`)? -> PLACED_OFF_TARGET."""
        if rest is None:
            return
        got = current_rest(T[:3, :3], obj)
        same = got == rest.name or (got or "").rstrip("+-") == rest.name.rstrip("+-")   # x+ / x-: same footprint
        if not same:
            label = rest_poses(obj)[got].label if got else "leaning (on no face)"
            raise Abort(PickPlace.Result.PLACED_OFF_TARGET, f"it ended {label}, not {rest.label}")

    def _placed_position(self, label, place_xy, obj, rest=None):
        """Where did the object end up -> ([x, y, z_top], source). 6-D pose if available (also catches an object
        that fell over, or one turned onto the wrong face), else the detector."""
        if self.get_parameter("grasp_source").value == "ground_truth":
            time.sleep(0.5)
            T = self._ground_truth_pose()
            self._face_check(T, obj, rest)
            tilt = tilt_deg(T[:3, :3], obj)
            if rest is None and min(tilt, 180 - tilt) > self.get_parameter("max_place_tilt_deg").value:
                raise Abort(PickPlace.Result.PLACED_OFF_TARGET, f"object tilted {tilt:.0f} deg after placing")
            return [T[0, 3], T[1, 3], T[2, 3] + obj["dims_m"][2] / 2], "ground truth (sim)"
        if self.get_parameter("grasp_source").value != "detector":
            T, _ = self._fresh_pose(time.time(), 3.0)       # after the release the pose node re-finds it at the goal
            if T is not None and np.hypot(T[0, 3] - place_xy[0], T[1, 3] - place_xy[1]) < 0.05:
                self._face_check(T, obj, rest)
                tilt = tilt_deg(T[:3, :3], obj)
                if rest is None and min(tilt, 180 - tilt) > self.get_parameter("max_place_tilt_deg").value:
                    raise Abort(PickPlace.Result.PLACED_OFF_TARGET, f"object tilted {tilt:.0f} deg after placing")
                return [T[0, 3], T[1, 3], T[2, 3] + obj["dims_m"][2] / 2], "6-D pose"
        found = self.robot.detect(label) or []
        if not found:
            raise Abort(PickPlace.Result.PLACED_OFF_TARGET, "object not found after placing (fell over / out of view?)")
        p = min((d[0] for d in found), key=lambda c: np.hypot(c[0] - place_xy[0], c[1] - place_xy[1]))
        return [p[0], p[1], p[2] + obj["dims_m"][2] / 2], "detector"

    def _grasp_event(self, state, T_tcp_obj=None, T_base_obj=None):
        """Tell the pose node what the hand is doing: attached (+ grasp offset), released (+ where it was put),
        lost (dropped / unknown: search again)."""
        d = {"state": state, "object": self.current_obj, "tcp_frame": self.robot_cfg["frames"]["tcp"],
             "base_frame": self.robot_cfg["frames"]["base"]}
        if T_tcp_obj is not None:
            d["T_tcp_obj"] = np.asarray(T_tcp_obj).tolist()
        if T_base_obj is not None:
            d["T_base_obj"] = np.asarray(T_base_obj).tolist()
        self.pub_grasp.publish(String(data=json.dumps(d)))

    # --- close look with the wrist camera -------------------------------------------------------------------------
    def _close_look(self, *a, **kw):
        t = time.time()
        try:
            return self._close_look_inner(*a, **kw)
        finally:
            self.trial["s_look"] += time.time() - t

    def _close_look_inner(self, handle, attempt, g, plan, T_obj, obj, grasp_cfg, lin, force=False):
        """At the pre-grasp pose: refine the object pose with the wrist camera and correct the grasp.
        -> (grasp, plan, T_base_obj), unchanged when the close look is off, fails or disagrees too much."""
        disputed = getattr(self, "disputed", None)
        if self.wrist_cfg is None or not (self.get_parameter("wrist_refine").value or force or disputed):
            return g, plan, T_obj

        def unsettled(why):
            # The 6-D pose and the detector disagreed, the detector's estimate was taken, and the close look can't
            # confirm it. Grasping anyway knocked a box off the table (detector 28 mm off near the view's edge, the
            # 6-D pose was right): stop instead and say why.
            raise Abort(PickPlace.Result.UNCERTAIN,
                        f"the overhead estimates disagree ({disputed}) and the close look couldn't settle it ({why}): "
                        f"not grasping blind -- scan again, or check the camera calibration")
        self._fb(handle, "LOOK", attempt, "wrist camera close-up: refining the object pose"
                 + (" (retry: looking even though wrist_refine is off)" if force and
                    not self.get_parameter("wrist_refine").value else ""))
        if not self.refine_cli.wait_for_service(timeout_sec=1.0):
            if disputed:
                unsettled("no close look available")
            self._fb(handle, "LOOK", attempt, "no /perception/refine_pose (pose_estimation:=true?): first estimate kept")
            return g, plan, T_obj
        look, dist = self._look_pose(T_obj, obj, plan)
        if look is not None:
            self._fb(handle, "LOOK", attempt, f"backing off: the whole object fits the wrist view at {dist:.2f} m")
            try:
                self._move(PickPlace.Result.MOTION_FAILED, "move to the look pose", self.robot.move_pose,
                           look.position, look.quat_xyzw)
            except Abort as e:                   # e.g. an obstacle in the way: the look is optional, the task isn't
                if disputed:
                    unsettled(f"look pose not reachable: {e}")
                self._fb(handle, "LOOK", attempt, f"look pose not reachable ({e}): first estimate kept")
                q_pre, _ = self._plan_approach([(plan.pregrasp, plan.grasp, plan)], lin, "pick")
                self._move(PickPlace.Result.UNREACHABLE, "back to pre-grasp", self.robot.move_joints, q_pre)
                return g, plan, T_obj
        req = RefinePose.Request(timeout_s=2.0)
        req.initial.header.frame_id = self.robot_cfg["frames"]["base"]
        req.initial.pose.position.x, req.initial.pose.position.y, req.initial.pose.position.z = map(float, T_obj[:3, 3])
        o = req.initial.pose.orientation
        o.x, o.y, o.z, o.w = map(float, matrix_to_quat(T_obj[:3, :3]))
        res = self.robot.wait(self.refine_cli.call_async(req), 10.0)

        def keep(why):                     # first estimate; back to its pre-grasp if we moved away for the look
            self._event("close_look_rejected", "warn", attempt=attempt, why=why, disputed=bool(disputed))
            if disputed:
                unsettled(why)
            self._fb(handle, "LOOK", attempt, f"{why}: first estimate kept")
            if look is not None:
                q_pre, _ = self._plan_approach([(plan.pregrasp, plan.grasp, plan)], lin, "pick")
                self._move(PickPlace.Result.UNREACHABLE, "back to pre-grasp", self.robot.move_joints, q_pre)
            return g, plan, T_obj
        if res is None or not res.success:
            return keep(f"close look failed ({res.message if res else 'timeout'})")
        q, t = res.pose.pose.orientation, res.pose.pose.position
        T_ref = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [t.x, t.y, t.z])
        deg = rotation_error_deg(T_obj[:3, :3], T_ref[:3, :3], obj)      # a can's spin about its axis doesn't count
        if (res.correction_m > self.get_parameter("max_refine_correction_m").value
                or deg > self.get_parameter("max_refine_correction_deg").value):
            return keep(f"close look disagrees too much ({res.correction_m * 1000:.0f} mm, {deg:.0f} deg)")
        grasps, why = grasps_from_pose(T_ref, obj, [plan.grasp.yaw], self.robot.tcp_yaw(),
                                       self.get_parameter("max_grasp_tilt_deg").value,
                                       self.robot_cfg["gripper"]["max_width_m"])
        grasps = shift_for_fixed_jaw(grasps, self.robot.gripper_model.held_offset_y,
                                      self.robot.gripper_model.approach_clearance)
        if not grasps:
            return keep(f"refined pose has no grasp ({why})")
        # the SAME grasp at the refined pose (nearest yaw): a square brick has two, 90 deg apart, and the first of the
        # list may be the one the pickability check ruled out (seen: a 2x2 with a cube 1 cm beside it -- the clear
        # grasp was chosen, the close look switched to the blocked one and the pick was refused)
        g2 = min(grasps, key=lambda c: abs(wrap_angle(tcp_yaw_of(c.T_base_tcp[:3, :3]) - plan.grasp.yaw)))
        plan2 = g2.plan(self.robot_cfg["gripper"]["max_width_m"], grasp_cfg["force_n"], **self._heights)
        # the corrected approach, planned like the first one (free move above + a checked straight line down). Not
        # plannable (seen on the SO-101: a corrected brick grasp near the base -> GOAL_IN_COLLISION): the first one,
        # already checked, is a few mm off at most -- better than giving up (unless the estimates disagreed)
        self._target_on(T_ref, obj)                # the target box where the close look put it, BEFORE planning near it
        time.sleep(0.3)                             # MoveIt filters its voxels from the next clouds
        try:
            q_pre, _ = self._plan_approach([(plan2.pregrasp, plan2.grasp, plan2)], lin, "pick (refined)")
        except Abort as e:
            self._target_on(T_obj, obj)             # (back to the first estimate's box if we keep that one)
            contacts = getattr(self.robot, "_last_contacts", "")
            if "<octomap>" in contacts:
                self._event("close_look_blocked", "warn", attempt=attempt, contacts=contacts)
                # the BETTER estimate says the jaws would go into something the camera sees beside the object: the
                # first estimate is just wrong there, not "a few mm off" (seen: a cube 1 cm beside a 2x4, the jaw
                # landed on the brick twice and pushed it 7 mm). Stop, don't grasp on the stale pose.
                raise Abort(PickPlace.Result.BLOCKED,
                            f"the close look puts the gripper into an obstacle beside the object ({contacts}): no room "
                            f"for the open jaws at the refined pose")
            return keep(f"corrected approach not plannable ({e})")
        self._move(PickPlace.Result.UNREACHABLE, "move to the corrected pre-grasp", self.robot.move_joints, q_pre)
        self.trial["corrections"].append([round(res.correction_m * 1000, 2), round(deg, 2)])
        self._fb(handle, "LOOK", attempt, f"refined: moved {res.correction_m * 1000:.1f} mm, {deg:.1f} deg "
                                          f"-> grasp corrected")
        return g2, plan2, T_ref

    def _staged_descent(self, handle, attempt, g, plan, T_obj, obj, grasp_cfg, lin):
        """closed_loop_approach: down to closed_loop_stage_m above the grasp, a fresh wrist look there, the last bit
        corrected. -> (g, plan, T_obj) to finish the approach with, or None after backing up (too far off / lost)."""
        R = self.robot
        h = float(self.get_parameter("closed_loop_stage_m").value)
        mid = np.asarray(plan.grasp.position, float) + [0.0, 0.0, h]
        self._move(PickPlace.Result.MOTION_FAILED, "staged approach", R.move_pose, mid, plan.grasp.quat_xyzw,
                   linear=True, scaling=lin)
        t = time.time()
        req = RefinePose.Request(timeout_s=2.0)
        req.initial.header.frame_id = self.robot_cfg["frames"]["base"]
        req.initial.pose.position.x, req.initial.pose.position.y, req.initial.pose.position.z = map(float, T_obj[:3, 3])
        o = req.initial.pose.orientation
        o.x, o.y, o.z, o.w = map(float, matrix_to_quat(T_obj[:3, :3]))
        res = R.wait(self.refine_cli.call_async(req), 10.0) if self.refine_cli.wait_for_service(timeout_sec=1.0) \
            else None
        self.trial["s_look"] += time.time() - t

        def back(why):
            self._event("staged_backoff", "warn", attempt=attempt, why=why)
            self._fb(handle, "APPROACH", attempt, f"staged look {h * 100:.0f} cm above: {why} -> backing up, retry")
            R.move_pose(plan.pregrasp.position, plan.pregrasp.quat_xyzw, linear=True, scaling=lin)
            return None
        if res is None or not res.success:
            return back(f"object not found ({res.message if res else 'no refine service'})")
        q, tt = res.pose.pose.orientation, res.pose.pose.position
        T_ref = make_transform(quat_to_matrix([q.x, q.y, q.z, q.w]), [tt.x, tt.y, tt.z])
        dxy = float(np.hypot(*(T_ref[:2, 3] - T_obj[:2, 3])))
        deg = rotation_error_deg(T_obj[:3, :3], T_ref[:3, :3], obj)
        self.trial["corrections"].append([round(dxy * 1000, 2), round(deg, 2), "staged"])
        if dxy > self.get_parameter("closed_loop_max_m").value or deg > self.get_parameter("closed_loop_max_deg").value:
            return back(f"{dxy * 1000:.1f} mm / {deg:.1f} deg off: more than the jaws allow")
        if dxy <= self.get_parameter("closed_loop_correct_m").value:
            self._fb(handle, "APPROACH", attempt, f"staged look: {dxy * 1000:.1f} mm off, on target")
            return g, plan, T_obj
        grasps, _ = grasps_from_pose(T_ref, obj, [plan.grasp.yaw], R.tcp_yaw(),
                                     self.get_parameter("max_grasp_tilt_deg").value,
                                     self.robot_cfg["gripper"]["max_width_m"])
        grasps = shift_for_fixed_jaw(grasps, R.gripper_model.held_offset_y, R.gripper_model.approach_clearance)
        if not grasps:
            return back("no grasp at the refined pose")
        g2 = min(grasps, key=lambda c: abs(wrap_angle(tcp_yaw_of(c.T_base_tcp[:3, :3]) - plan.grasp.yaw)))
        plan2 = g2.plan(self.robot_cfg["gripper"]["max_width_m"], grasp_cfg["force_n"], **self._heights)
        self._target_on(T_ref, obj)
        self._fb(handle, "APPROACH", attempt, f"staged look: {dxy * 1000:.1f} mm / {deg:.1f} deg off -> corrected")
        return g2, plan2, T_ref

    def _look_pose(self, T_obj, obj, plan):
        """-> (TCP Pose to look from, camera distance) if the object doesn't fit the wrist view at the pre-grasp pose
        (or is closer than the camera's minimum depth range), else (None, distance). The camera is put on the line through the object's center along its own viewing
        direction (TCP orientation as at the pre-grasp), at the distance where the object's diameter fills look_fill
        of the vertical field of view."""
        w = self.wrist_cfg
        try:
            tf = self.robot.tf.lookup_transform(self.robot_cfg["frames"]["tcp"], w["frames"]["color_optical"],
                                                Time()).transform
        except Exception:  # noqa: BLE001
            return None, 0.0
        T_tcp_opt = make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                                   [tf.translation.x, tf.translation.y, tf.translation.z])
        vfov = 2 * np.arctan(w["height"] / 2 / w["fy"])
        diameter = float(np.linalg.norm(obj["dims_m"]))
        need = diameter / (2 * np.tan(vfov / 2) * self.get_parameter("look_fill").value)
        need = max(need, 1.2 * float(w.get("depth", {}).get("min_m", 0.0)))   # no depth closer than that (D405 7 cm)
        R_tcp = quat_to_matrix(plan.pregrasp.quat_xyzw)
        cam_at_pre = plan.pregrasp.position + R_tcp @ T_tcp_opt[:3, 3]
        dist_now = float(np.linalg.norm(T_obj[:3, 3] - cam_at_pre))
        if dist_now >= 0.9 * need:
            return None, dist_now
        view = R_tcp @ T_tcp_opt[:3, 2]                         # the camera's viewing direction (world)
        cam = T_obj[:3, 3] - need * view
        tcp = cam - R_tcp @ T_tcp_opt[:3, 3]
        return Pose(tcp, plan.pregrasp.quat_xyzw), need

    # --- map mode: continuous / frozen per task + change monitor ------------------------------------------------
    def _perceived(self):
        return not self.get_parameter("use_known_obstacles").value

    def _set_frozen(self, on: bool) -> bool:
        """Perceived: the obstacle cloud node stops/resumes feeding MoveIt (and starts/stops comparing)."""
        if not self.freeze_cli.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("/perception/map_freeze not available (obstacle_cloud_node running?)")
            return False
        res = self.robot.wait(self.freeze_cli.call_async(SetBool.Request(data=on)), 5.0)
        return bool(res and res.success)

    def _open_width(self, grasp_cfg) -> float:
        """How wide to open for this object: fully (default), or object width + gripper.open_margin_m per side. A small
        cell (SO-101) needs the latter: its jaw swings out sideways, and a fully open jaw hit a box next to the goal."""
        g = self.robot_cfg["gripper"]
        if "open_margin_m" not in g:
            return float(g["max_width_m"])
        return float(min(g["max_width_m"], grasp_cfg["width_m"] + 2 * float(g["open_margin_m"])))

    def _check_calibration(self):
        c = self.calibration
        if c.get("state") != "DRIFT":
            return
        what = f"the fixed camera moved since its calibration ({c.get('px')} px ~ {c.get('mm')} mm at the AprilTag)"
        if self.get_parameter("calibration_policy").value == "stop":
            raise Abort(PickPlace.Result.CALIBRATION_DRIFT, f"{what}: recalibrate (tools/calibrate.py fixed)")
        self.get_logger().warn(f"{what}: running anyway (calibration_policy warn)")

    def _freeze_map(self, handle):
        """freeze_per_task: a fresh look at the scene with the arm at home, then the whole task plans against it."""
        if self.get_parameter("map_mode").value != "freeze_per_task":
            return
        settle = self.get_parameter("map_settle_s").value
        if self._perceived():
            # NOT cleared first: the live map still holds what the arm (at home) hides right now; clearing made those
            # regions unknown = free for the planner. Stale voxels are cleared continuously (far rays, cloud.py).
            time.sleep(settle)                                  # fresh frames of the arm-at-home view
            if not self._set_frozen(True):
                self.get_logger().warn("could not freeze the map: this task runs with continuous updates")
                return
        else:
            self._sync_obstacles()                              # the newest known layout, then hold it
        self.frozen, self.changes = True, (None, 0.0)
        policy = self.get_parameter("change_policy").value
        self._fb(handle, "HOME", detail=f"map frozen for this task (change policy: {policy})")

    def _unfreeze_map(self, settle=False):
        if not self.frozen:
            return
        self.frozen = False
        if self._perceived():
            self._set_frozen(False)
            if settle:
                time.sleep(self.get_parameter("map_settle_s").value)   # let live frames update the Octomap
        else:
            self._sync_obstacles()                              # apply what arrived during the task

    def _refresh_map(self):
        """Update the snapshot from the live camera WITHOUT clearing (voxels hidden behind the arm stay)."""
        if self._perceived():
            self._set_frozen(False)
            time.sleep(self.get_parameter("map_settle_s").value)
            self._set_frozen(True)
        else:
            self.frozen = False
            self._sync_obstacles()
            self.frozen = True
        self.changes = (None, 0.0)

    def _scene_change(self):
        """-> description of what changed since the snapshot, or None."""
        if self._perceived():
            report, t = self.changes
            if not report or not report.get("frozen") or not report.get("changed") or time.time() - t > 2.0:
                return None
            parts = [f"{len(report[k])} {k}: " + "; ".join(
                         f"{c['voxels']} voxels at {c['center']} size {c['size']}" for c in report[k])
                     for k in ("appeared", "gone") if report[k]]
            return ", ".join(parts) or "changed"
        if self.known is not None and json.dumps(self.known) != self.applied[0]:
            return "the known obstacle layout changed"
        return None

    def _check_scene(self, handle, where, attempt):
        """Before a planned move in a frozen task: did the scene change? -> change_policy.
        Newly SEEN things (hidden when the snapshot was taken) are no change: they are always taken into the map."""
        if not self.frozen:
            return
        report, t = self.changes
        if (self._perceived() and report and report.get("incomplete") and not report.get("changed")
                and time.time() - t < 2.0):
            seen = "; ".join(f"{c['voxels']} voxels at {c['center']}" for c in report["newly_seen"])
            self._event("scene_change", where=where, decision="map_completed", what=seen)
            self._fb(handle, where, attempt, f"map completed: newly visible ({seen}) -> taking it into the map")
            self._refresh_map()
            return
        what = self._scene_change()
        if what is None:
            return
        policy = self.get_parameter("change_policy").value
        self._event("scene_change", "warn", where=where, decision=policy, what=str(what))
        if policy == "stop":
            raise Abort(PickPlace.Result.SCENE_CHANGED, f"scene changed before {where}: {what}")
        if policy == "refresh":
            self._fb(handle, where, attempt, f"scene changed ({what}) -> refreshing the map, then planning")
            self._refresh_map()
        else:
            self._fb(handle, where, attempt, f"scene changed ({what}) -> WARNING: still planning with the frozen map")

    # --- helpers --------------------------------------------------------------------------------------------
    def _yaws(self, yaw, symmetric):
        """Grasp yaws to try, best first. A parallel gripper is 180-deg symmetric; a rotationally symmetric
        object (the can) accepts ANY yaw, so also try turning the hand: with a box beside the object, the open hand
        (~27 cm long in MoveIt, incl. franka_description's 3 cm safety margin) may only fit one way."""
        return grasp_yaws(yaw, self.robot.tcp_yaw(), symmetric)

    def _retreat_turned(self, handle, attempt, place, lin, back_m=0.10, up_m=0.08):
        """Out of a turned (horizontal) hand's placement: slide the open hand BACK along its own axis first, then up.
        Straight up from there had no IK along the line (seen), and the recovery that followed moved the arm with
        the open hand still around the box: it knocked the upside-down box over."""
        R = self.robot
        z = quat_to_matrix(place.quat_xyzw)[:, 2]                  # the hand's approach axis (horizontal-ish)
        back = np.asarray(place.position, float) - back_m * z
        self._move(PickPlace.Result.MOTION_FAILED, "retreat (back out of the object)", R.move_pose, back,
                   place.quat_xyzw, linear=True, scaling=lin)
        code = R.move_pose(back + [0.0, 0.0, up_m], place.quat_xyzw, linear=True, scaling=lin)
        if code != SUCCESS_CODE:                                  # clear of the object already: home goes free-space
            self._fb(handle, "RETREAT", attempt, f"lift after backing out not possible ({error_name(code)}): "
                                                 f"going home from here")

    def _plan_approach(self, *a, **kw):
        t = time.time()
        try:
            return self._plan_approach_inner(*a, **kw)
        finally:
            if getattr(self, "trial", None) is not None:
                self.trial["s_plan"] += time.time() - t

    def _plan_approach_inner(self, candidates, lin, what, tries=2, straight=True):
        """Plan 'free-space move to `above`, then straight line to `target`' as ONE unit, nothing moves yet.
        candidates: [(above, target, payload)], e.g. one per grasp yaw, best first.
        Returns (arm configuration to go to above the target, payload of the candidate that worked).

        Why: the straight line (Pilz LIN) has no collision avoidance; it follows the TCP down and the arm's
        posture follows from IK. With obstacles next to the target, the forearm can sweep into one on the way down
        (seen: fr3_link5 vs the tower) even though the pose above was fine. The FR3 has 7 joints, so each TCP pose
        has many arm configurations (elbow higher/lower...), and OMPL samples a random one. So: plan the move
        above, take the configuration it ends in, check the straight line FROM there; if blocked, sample again."""
        R, last = self.robot, ""
        for i, (above, target, payload) in enumerate(candidates):
            for k in range(tries):
                if not straight:
                    # A turned (horizontal) hand: choose the arm configuration AT THE TARGET first (MoveIt samples a
                    # collision-free one), then the one above it NEAR it, then check the joint-space line between.
                    # Above-first kept whatever branch the planner happened to pick up there, and from it every way
                    # down -- a Cartesian line (Pilz LIN), our IK near it -- put the forearm (fr3_link5/6) into the
                    # desk; a free-space (OMPL) descent once swung joint 1 by 155 deg and threw the box behind the
                    # robot.
                    code, q_t = R.plan_pose(target.position, target.quat_xyzw, plan_only=True)
                    q_a = R.ik_near(above.position, above.quat_xyzw, q_t) if code == SUCCESS_CODE and q_t is not None                         else None
                    if q_a is None:
                        last = f"{what} not reachable: {error_name(code) if code != SUCCESS_CODE else 'no IK above it'}"
                        continue
                    code = R.plan_joints(q_t, start_q=q_a, ptp=True)            # down: a joint-space line
                    if code == SUCCESS_CODE:
                        code = R.plan_joints(q_a, ptp=False)                    # and the arm can get above it
                    if code == SUCCESS_CODE:
                        self.last_q_target = q_t                # the turned descent goes to exactly these joints
                        if i or k:
                            self.get_logger().info(f"{what}: approach found with candidate {i + 1}, try {k + 1}")
                        return q_a, payload
                    last = f"{what}: no way down to it ({error_name(code)})"
                    self.get_logger().warn(f"{what}: turned approach blocked (candidate {i + 1}/{len(candidates)}, "
                                           f"configuration {k + 1}/{tries}: {error_name(code)})")
                    continue
                code, q_above = R.plan_pose(above.position, above.quat_xyzw, plan_only=True)
                if code != SUCCESS_CODE or q_above is None:
                    last = f"{what} not reachable: {error_name(code)}"
                    if error_name(code) == "FAILURE" and k + 1 < tries:     # a sampled path that failed validation
                        continue                                            # (seen: a turned hand low over the desk)
                    break                                                   # no IK / invalid goal: next candidate
                path_above = list(R.last_path)
                lift = self._lift_of(payload) if what == "pick" else None
                if what.startswith("pick"):
                    self._target_contact(True)          # checking the line down to the grasp: the jaws reach around it
                try:
                    code, q_grasp = R.plan_pose(target.position, target.quat_xyzw, linear=True, scaling=lin,
                                                start_q=q_above)
                    path_down = list(R.last_path)
                    path_up = []
                    if code == SUCCESS_CODE and lift is not None and q_grasp is not None:
                        # and the LIFT from where that line ends: the WHOLE pick is planned before anything moves
                        code_up, _ = R.plan_pose(lift.position, lift.quat_xyzw, linear=True, scaling=lin,
                                                 start_q=q_grasp)
                        path_up = list(R.last_path)
                        if code_up != SUCCESS_CODE:
                            code = code_up
                            last = f"{what}: the lift after the grasp is blocked ({error_name(code_up)})"
                            self.get_logger().warn(f"{what}: lift blocked (candidate {i + 1}/{len(candidates)}, "
                                                   f"configuration {k + 1}/{tries}: {error_name(code_up)})")
                            continue
                finally:
                    if what.startswith("pick"):
                        self._target_contact(False)
                if code == SUCCESS_CODE:
                    self.planned_segments[what] = [(f"{what}: above", path_above), (f"{what}: down", path_down)] + \
                        ([(f"{what}: lift", path_up)] if path_up else [])
                    if i or k:
                        self.get_logger().info(f"{what}: approach found with candidate {i + 1}, try {k + 1}")
                    return q_above, payload
                last = f"{what}: straight approach blocked for every grasp yaw / arm configuration ({error_name(code)})"
                self.get_logger().warn(f"{what}: straight approach blocked (candidate {i + 1}/{len(candidates)}, "
                                       f"configuration {k + 1}/{tries}: {error_name(code)})")
        raise Abort(PickPlace.Result.UNREACHABLE, last)

    def _setup_scene(self, table_z):
        """The table as a collision object. Its top is placed 4 mm BELOW the real surface: the robot base sits
        on the table, and touching it would make every start state 'in collision'."""
        t = self.scene_cfg["table"]
        center = list(t["center"])
        center[2] = table_z - 0.004 - t["size"][2] / 2
        self.robot.add_box("table", center, t["size"])

    def _on_known(self, msg):
        if not self.get_parameter("use_known_obstacles").value:
            return                                   # Layer B: the robot must SEE the boxes (Octomap)
        self.known = json.loads(msg.data)["boxes"]
        self._sync_obstacles()

    def _sync_obstacles(self):
        """Make MoveIt's obstacles match the latest known layout: add/replace the current boxes (inflated by
        padding_m on every side as a safety margin), remove the ones that are gone. Never raises (it runs in a
        subscription callback); if MoveIt isn't up yet, the next message (1 Hz) retries."""
        with self.scene_lock:
            if self.known is None or json.dumps(self.known) == self.applied[0]:
                return
            if self.frozen:                   # freeze_per_task: keep the snapshot; _check_scene reports the change
                return
            pad = self.scene_cfg["obstacles"]["padding_m"]
            try:
                ids = set()
                for b in self.known:
                    name = f"obstacle_{b['id']}"
                    self.robot.add_box(name, b["center"], [s + 2 * pad for s in b["size"]], b["yaw"])
                    ids.add(name)
                self.robot.remove_objects(sorted(self.applied[1] - ids))
                self.applied = (json.dumps(self.known), ids)
                self.get_logger().info(f"planning scene: {len(ids)} known obstacle(s) {sorted(ids)}")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"could not sync obstacles yet: {e}")

    # --- pickability: is there room for the open hand at all? (pnp_core/pickability.py) ------------------------------
    def _on_table_points(self, msg):
        pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.table_points = (time.time(), np.asarray(pts, float).reshape(-1, 3))

    def _pickable_grasps(self, handle, attempt, grasps, T_obj, obj, sym, since):
        """The grasps whose open jaws, hand and descent are clear of everything else on the table (no target cut-out);
        none -> BLOCKED with the reason, the arm hasn't moved. Without /perception/table_points (obstacle_source:=known)
        nothing to check against: all grasps kept."""
        end = time.time() + 2.0
        while (self.table_points is None or self.table_points[0] < since) and time.time() < end:
            time.sleep(0.05)
        if self.table_points is None or self.table_points[0] < since:
            self._fb(handle, "DETECT", attempt, "no fresh /perception/table_points: pickability not checked")
            return grasps
        r = pick_check(self.table_points[1], obj, T_obj, self.robot_cfg["gripper"], self.robot.gripper_model,
                       self.robot.tcp_yaw(), self._yaws(0.0, sym), self.get_parameter("max_grasp_tilt_deg").value,
                       approach_height=self._heights["approach_height"],
                       label=obj.get("label") or obj["yolo_class"].replace("_", " "),
                       inflate=check_inflate(self.scene_cfg),          # the same as the workcell page
                       flips=bool(self.get_parameter("grasp_scoring").value))
        if not r["pickable"]:
            raise Abort(PickPlace.Result.BLOCKED, r["why"])
        if len(r["clear"]) < len(r["results"]):
            self._fb(handle, "DETECT", attempt, f"{len(r['clear'])} of {len(r['results'])} grasps have room for the hand "
                                                f"(an obstacle is near the others)")
        self.grasp_clearance = [r["results"][i]["clearance"] for i in r["clear"]]
        return [r["grasps"][i] for i in r["clear"]]

    # --- the target (and the placed object) as MoveIt world objects --------------------------------------------------
    def _preview(self, handle, g, place_candidates, lin, rest):
        """The pick is planned (pre-grasp, down, lift: _plan_approach); now the place from where the lift ends, plan
        only, and the whole path to /display_planned_path. Blocked -> Abort with why; feasible -> preview_message."""
        R, segs = self.robot, list(self.planned_segments.get("pick", []))
        q_lift = segs[-1][1][-1] if segs and segs[-1][1] else None
        try:
            if rest is None and q_lift is not None:
                why = "no place candidate"
                for pre, place, _ in place_candidates(g.T_tcp_obj, g.width_m):
                    code, q_pp = R.plan_pose(pre.position, pre.quat_xyzw, plan_only=True, start_q=q_lift)
                    p_above = list(R.last_path)
                    if code != SUCCESS_CODE or q_pp is None:
                        why = f"above the goal not reachable from the lift ({error_name(code)})"
                        continue
                    code, _ = R.plan_pose(place.position, place.quat_xyzw, linear=True, scaling=lin, start_q=q_pp)
                    if code == SUCCESS_CODE:
                        segs += [("place: above", p_above), ("place: down", list(R.last_path))]
                        why = ""
                        break
                    why = f"the straight line down at the goal is blocked ({error_name(code)})"
                if why:
                    self._publish_path(segs)
                    raise Abort(PickPlace.Result.UNREACHABLE, f"preview: the pick works, the place doesn't: {why}")
            self._publish_path(segs)
            names = ", ".join(n for n, _ in segs)
            self.preview_message = (f"preview: feasible ({names}; {sum(len(p) for _, p in segs)} waypoints, shown in "
                                    f"RViz 'Planned motion') -- press Pick & place to run it")
            self._fb(handle, "PREVIEW", 0, self.preview_message)
        finally:
            self._scene_objects_off()

    def _publish_path(self, segs):
        """Segments [(name, [arm joint positions])] -> one trajectory on /display_planned_path (timed by joint travel)."""
        pts = [q for _, path in segs for q in path]
        if not pts:
            return
        joints = list(self.robot_cfg["arm"]["joints"])
        dt = DisplayTrajectory()
        dt.trajectory_start.joint_state.name = joints
        dt.trajectory_start.joint_state.position = [float(v) for v in pts[0]]
        rt = RobotTrajectory()
        rt.joint_trajectory.joint_names = joints
        t, prev = 0.0, np.asarray(pts[0], float)
        for q in pts:
            q = np.asarray(q, float)
            t += max(0.04, float(np.max(np.abs(q - prev))) / 0.5)     # ~0.5 rad/s: a watchable animation
            prev = q
            rt.joint_trajectory.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in q], time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))))
        dt.trajectory = [rt]
        self.pub_display.publish(dt)

    @staticmethod
    def _lift_of(payload):
        """The lift pose of a pick candidate's payload: a GraspPlan, or (grasp, GraspPlan)."""
        for p in (payload, *(payload if isinstance(payload, tuple) else ())):
            if hasattr(p, "lift"):
                return p.lift
        return None

    def _hand(self):
        return list(self.robot_cfg.get("hand_links", [self.robot_cfg["frames"]["tcp"]]))

    def _target_on(self, T_obj, obj):
        """The pick target as a collision object at T_obj (6-D pose or the detector's position + yaw: the catalog shape
        either way), grown by target_padding_m. The first time in a task: wait target_clear_s so MoveIt has filtered
        its voxels out of the Octomap before anything is planned near it."""
        self.robot.add_object(TARGET_ID, obj, T_obj, self.get_parameter("target_padding_m").value)
        if not self.target_in_scene:
            self.target_in_scene = True
            self.robot.allow_contact(TARGET_ID, self._hand(), False)
            time.sleep(self.get_parameter("target_clear_s").value)

    def _target_contact(self, allowed):
        if self.target_in_scene:
            self.robot.allow_contact(TARGET_ID, self._hand(), allowed)

    def _target_off(self):
        if self.target_in_scene:
            self.robot.allow_contact(TARGET_ID, self._hand(), False)
            self.robot.remove_objects([TARGET_ID])
            self.target_in_scene = False

    def _scene_objects_off(self):
        """No target / placed object in MoveIt's world, no contact allowed with them (a task's start and end, recovery)."""
        for name in (TARGET_ID, PLACED_ID):
            self.robot.allow_contact(name, self._hand(), False)
        self.robot.remove_objects([TARGET_ID, PLACED_ID])
        self.target_in_scene = False

    def _safe_recover(self):
        """After a failure: if we're holding the object, put it back down where we picked it (never just open
        the gripper in mid-air); then open, detach, go home. Best effort, never raises."""
        R, open_w = self.robot, self.robot_cfg["gripper"]["max_width_m"]
        R.free_scaling = self.robot_cfg["arm"]["speed_profiles"]["dev"]["free"]   # recover gently
        steps = []
        if self.holding is not None:
            p = self.holding
            self.get_logger().warn("recovery: putting the object back where it was picked")
            self._event("recovery", "warn", holding=True)
            steps += [lambda: self._require(R.move_pose(p.pregrasp.position, p.pregrasp.quat_xyzw), "back above pick"),
                      lambda: self._require(R.move_pose(p.grasp.position, p.grasp.quat_xyzw, linear=True,
                                                        scaling=self.robot_cfg["arm"]["speed_profiles"]["dev"]["linear"]), "down")]
        steps += [lambda: R.gripper(open_w, 30.0),
                  lambda: R.detach(GRASPED_ID)]
        if self.holding is not None and self.held_from is not None:
            T_back = self.held_from
            steps += [lambda: self._grasp_event("released", T_base_obj=T_back)]   # put back where it was
        if self.holding is not None:
            p = self.holding
            steps += [lambda: R.move_pose(p.pregrasp.position, p.pregrasp.quat_xyzw, linear=True)]
        steps += [lambda: R.move_joints(self.robot_cfg["arm"]["home"]), self._scene_objects_off]
        for step in steps:
            try:
                step()
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"recovery step failed: {e}")
        self.holding = None

    @staticmethod
    def _require(code, what):
        if code != SUCCESS_CODE:
            raise RuntimeError(f"{what}: {error_name(code)}")


def main():
    rclpy.init()
    node = PickPlaceNode()
    ex = MultiThreadedExecutor(num_threads=6)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
