"""Thin, blocking wrappers around the standard ROS 2 interfaces the pick & place logic needs.

    MoveIt 2      /move_action            (moveit_msgs/MoveGroup)          plan + execute (OMPL or Pilz LIN)
    MoveIt 2      /apply_planning_scene   (moveit_msgs/ApplyPlanningScene)  table, obstacles, attach/detach
    ros2_control  <gripper action>        (control_msgs/GripperCommand)     open / close
    perception    /perception/detect      (pnp_interfaces/DetectObjects)    fresh 3D detections
    state         /joint_states, TF

Everything here is a standard interface that exists unchanged on the real robot, so the state machine on
top never needs to know whether it drives Isaac Sim or a real FR3.

Threading: these calls BLOCK, so they must run in a thread other than the one spinning the node (the action
server's execute callback in a MultiThreadedExecutor). They wait by polling futures, never by spinning.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Pose
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (AllowedCollisionEntry, AttachedCollisionObject, BoundingVolume, CollisionObject,
                             Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes, OrientationConstraint,
                             PlanningScene, PlanningSceneComponents, PositionConstraint)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity
from pnp_interfaces.srv import DetectObjects
from std_srvs.srv import Empty, SetBool
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from rclpy.time import Time
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from pnp_core import robot_description as rdesc
from pnp_core.catalog import repo_root
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix
from pnp_core.grasp import R_DOWN
from pnp_core.gripper import GripperModel
from pnp_core.object_grasp import object_extents, rotation_between, up_axis
from pnp_core.urdf_kin import Chain


class Paused(Exception):
    """A move was stopped for a hand in the workspace; the workspace is clear again: resend / resume it."""

ERROR_NAMES = {v: k for k, v in MoveItErrorCodes.__dict__.items() if k.isupper() and isinstance(v, int)}


class RobotInterface:
    def __init__(self, node, robot_cfg: dict):
        self.node, self.cfg = node, robot_cfg
        cb = ReentrantCallbackGroup()
        self.js = {}
        node.create_subscription(JointState, "/joint_states", self._js, 20, callback_group=cb)
        self.tf = Buffer()
        TransformListener(self.tf, node)
        self.move = ActionClient(node, MoveGroup, "/move_action", callback_group=cb)
        self.grip = ActionClient(node, GripperCommand, robot_cfg["gripper"]["action"], callback_group=cb)
        self.traj = ActionClient(node, FollowJointTrajectory,
                                 f"/{robot_cfg['arm']['controller']}/follow_joint_trajectory", callback_group=cb)
        self.scene = node.create_client(ApplyPlanningScene, "/apply_planning_scene", callback_group=cb)
        self.get_scene = node.create_client(GetPlanningScene, "/get_planning_scene", callback_group=cb)
        # a hand in the workspace (pnp_perception/hand_detector_node, /safety/hand): a running arm move is stopped and
        # resumed once the workspace has been clear for hand_clear_s (a convenience stop for the SO-101's low forces,
        # NOT a safety system). Nothing published -> nothing pauses.
        self.hand = (False, 0.0)
        self.hand_pause = True
        self.hand_clear_s, self.hand_max_wait_s = 1.0, 120.0
        self.pause_listener = None                # callable(state: "paused" | "resumed", info: dict)
        self.pause_poll = None                    # callable, polled while paused (may raise: e.g. the task's cancel)
        node.create_subscription(String, "/safety/hand", self._on_hand, 10, callback_group=cb)
        # the REAL SO-101 (pnp_manipulation/so101_bridge): it starts limp (torque off) and switches torque off after
        # bus errors; every goal that moves something switches it on first (PNP_AUTO_TORQUE=0: never). Nothing
        # published (the sim) -> nothing happens
        self.bridge = (None, 0.0)                 # (enabled, time of the last /so101_bridge/status)
        node.create_subscription(String, "/so101_bridge/status", self._on_bridge, 10, callback_group=cb)
        self.torque_cli = node.create_client(SetBool, "/so101_bridge/enable", callback_group=cb)
        self.last_path = []           # arm joint waypoints of the last planned segment (MoveIt's or a straight line)
        self.validity = node.create_client(GetStateValidity, "/check_state_validity", callback_group=cb)
        self.detect_cli = node.create_client(DetectObjects, "/perception/detect", callback_group=cb)
        self.clear_octomap_cli = node.create_client(Empty, "/clear_octomap", callback_group=cb)
        self.base, self.tcp = robot_cfg["frames"]["base"], robot_cfg["frames"]["tcp"]
        self.free_scaling = robot_cfg["arm"]["velocity_scaling"]   # default speed for moves (set per task)
        self.gripper_model = GripperModel(robot_cfg["gripper"])
        # arm.ik: numeric -> pose goals are solved here (pnp_core/urdf_kin.py) into joint goals for MoveIt: arms with
        # fewer than 6 joints (SO-101), where MoveIt's KDL plugin can't solve full 6-D pose goals
        self.chain, self._last_contacts = None, ""
        if robot_cfg["arm"].get("ik") == "numeric":
            urdf_xml = rdesc.urdf(robot_cfg, repo_root(), "mock")
            self.chain = Chain(urdf_xml, self.base, self.tcp).clamp_limits(
                robot_cfg["arm"]["position_limits"], robot_cfg["arm"].get("limit_margin", 0.0) + 0.005)
        # IK NEAR a given arm configuration, for any robot (ik_near): the numeric chain, or one built from the
        # published /robot_description (the FR3, whose MoveIt uses KDL)
        self.ik_chain = self.chain
        if self.ik_chain is None:
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            node.create_subscription(String, "/robot_description", self._on_description, latched)

    def _on_description(self, msg):
        if self.ik_chain is None:
            ch = Chain(msg.data, self.base, self.tcp)
            if self.cfg["arm"].get("position_limits"):
                ch.clamp_limits(self.cfg["arm"]["position_limits"], 0.005)
            self.ik_chain = ch

    def ik_near(self, xyz, quat_xyzw, seed, max_jump=1.2):
        """Joint values for a TCP pose CLOSE to the configuration `seed` (the same arm branch), or None. MoveIt's
        KDL solver missed solutions for a turned (horizontal) hand that this finds from a nearby seed."""
        if self.ik_chain is None:
            return None
        q, pe, ae = self.ik_chain.ik(xyz, target_R=quat_to_matrix(quat_xyzw), q0=np.asarray(seed, float), seeds=1)
        if q is None or pe > 0.002 or ae > 1.0 or np.max(np.abs(np.asarray(q) - np.asarray(seed))) > max_jump:
            return None
        return q

    def plan_joints(self, q, start_q=None, ptp=True, scaling=None):
        """Plan only (nothing moves): can the arm go from start_q (default: where it is) to joints q?"""
        c = Constraints()
        c.joint_constraints = [JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005,
                                               tolerance_below=0.005, weight=1.0)
                               for j, v in zip(self.cfg["arm"]["joints"], q)]
        pipeline, planner = ("pilz_industrial_motion_planner", "PTP") if ptp else ("ompl", "")
        return self._run_full(self._request(c, pipeline, planner, scaling or self.free_scaling, True, start_q))[0]

    # --- low level ---------------------------------------------------------------------------------------
    def _js(self, msg):
        self.js.update(zip(msg.name, msg.position))

    @staticmethod
    def wait(fut, timeout):
        end = time.time() + timeout
        while not fut.done():
            if time.time() > end:
                return None
            time.sleep(0.01)
        return fut.result()

    def _on_hand(self, msg):
        try:
            import json
            self.hand = (bool(json.loads(msg.data).get("present")), time.time())
        except Exception:  # noqa: BLE001
            pass

    def hand_present(self) -> bool:
        return self.hand_pause and self.hand[0] and time.time() - self.hand[1] < 0.6

    def wait_hand_clear(self, why="") -> float:
        """Block until no hand has been seen for hand_clear_s -> the seconds waited (0: none was there)."""
        if not self.hand_present():
            return 0.0
        t0, clear_since = time.time(), None
        if self.pause_listener:
            self.pause_listener("paused", {"why": why})
        self.node.get_logger().warn(f"PAUSED: a hand in the workspace ({why})")
        while True:
            if self.pause_poll:
                self.pause_poll()
            if time.time() - t0 > self.hand_max_wait_s:
                raise RuntimeError(f"a hand stayed in the workspace for {self.hand_max_wait_s:.0f} s")
            if self.hand_present():
                clear_since = None
            elif clear_since is None:
                clear_since = time.time()
            elif time.time() - clear_since >= self.hand_clear_s:
                break
            time.sleep(0.05)
        waited = time.time() - t0
        if self.pause_listener:
            self.pause_listener("resumed", {"why": why, "waited_s": round(waited, 1)})
        self.node.get_logger().info(f"resumed after {waited:.1f} s (the workspace is clear)")
        return waited

    def _on_bridge(self, msg):
        try:
            self.bridge = (bool(json.loads(msg.data).get("enabled")), time.time())
        except (ValueError, AttributeError):
            pass

    def ensure_torque(self) -> bool:
        """The real arm's servos hold + follow commands: switch torque on if the bridge reports it off. The bridge
        then holds the CURRENT pose, restarts the controllers and follows only commands near the measured pose
        (arm_gate_rad): no jump. -> False if it's off and couldn't be switched on."""
        on, t = self.bridge
        if on is None or time.time() - t > 3.0 or on or os.environ.get("PNP_AUTO_TORQUE", "1") == "0":
            return True
        if not self.torque_cli.wait_for_service(timeout_sec=5.0):
            self.node.get_logger().error("the arm's torque is OFF and /so101_bridge/enable isn't reachable")
            return False
        self.node.get_logger().warn("the arm's torque is OFF: switching it on (it holds where it is)")
        res = self.wait(self.torque_cli.call_async(SetBool.Request(data=True)), 10.0)
        end = time.time() + 5.0
        while time.time() < end and not self.bridge[0]:
            time.sleep(0.1)
        time.sleep(2.0)            # the bridge ignores commands 1.5 s while the controllers restart on the held pose
        return bool(res and res.success and self.bridge[0])

    def _action(self, client, goal, timeout, pausable=False):
        """Send an action goal, wait for its result. pausable (a move that really moves the arm): a hand in the
        workspace cancels it and, once it's clear again, raises Paused (the caller resends / resumes)."""
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"action server {client._action_name} not available")
        moves = not (isinstance(goal, MoveGroup.Goal) and goal.planning_options.plan_only)
        if moves and not self.ensure_torque():
            raise RuntimeError("the arm's servo torque is off (see /so101_bridge/status)")
        if pausable:
            self.wait_hand_clear(f"before {client._action_name}")
        handle = self.wait(client.send_goal_async(goal), 10.0)
        if handle is None or not handle.accepted:
            raise RuntimeError(f"{client._action_name}: goal rejected")
        fut = handle.get_result_async()
        end = time.time() + timeout
        while not fut.done():
            if time.time() > end:
                handle.cancel_goal_async()
                raise RuntimeError(f"{client._action_name}: no result within {timeout} s")
            if pausable and self.hand_present():
                self.wait(handle.cancel_goal_async(), 3.0)              # the controller holds where it is
                self.wait_hand_clear(f"during {client._action_name}")
                raise Paused()
            time.sleep(0.01)
        return fut.result().result

    def wait_ready(self, timeout=30.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if all(j in self.js for j in self.cfg["arm"]["joints"]) and self.move.server_is_ready():
                return True
            time.sleep(0.1)
        return False

    # --- state -------------------------------------------------------------------------------------------
    def arm_q(self):
        return np.array([self.js[j] for j in self.cfg["arm"]["joints"]])

    def gripper_width(self) -> float:
        """Opening between the fingers / jaws [m] from the gripper joint (pnp_core/gripper.py: per gripper type)."""
        return self.gripper_model.width(float(self.js.get(self.cfg["gripper"]["driven_joint"], 0.0)))

    def tcp_pose(self):
        t = self.tf.lookup_transform(self.base, self.tcp, Time()).transform
        q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        return np.array([t.translation.x, t.translation.y, t.translation.z]), quat_to_matrix(q)

    def tcp_yaw(self) -> float:
        """Yaw of a (roughly) top-down TCP about world z, in the convention of pnp_core.grasp.top_down_quat."""
        _, R = self.tcp_pose()
        Rz = R @ R_DOWN.T
        return float(np.arctan2(Rz[1, 0], Rz[0, 0]))

    # --- motion ------------------------------------------------------------------------------------------
    def _request(self, constraints, pipeline, planner, scaling, plan_only=False, start_q=None):
        req = MotionPlanRequest()
        req.group_name = self.cfg["arm"]["planning_group"]
        if start_q is not None:     # plan from a hypothetical arm configuration (only meaningful with plan_only)
            req.start_state.is_diff = True
            req.start_state.joint_state.name = list(self.cfg["arm"]["joints"])
            req.start_state.joint_state.position = [float(v) for v in start_q]
        req.pipeline_id, req.planner_id = pipeline, planner
        req.num_planning_attempts, req.allowed_planning_time = 5, 5.0
        # PNP_MAX_VELOCITY_SCALING (the Jetson's robot container: the real SO-101) caps EVERY move, whoever asks
        scaling = min(float(scaling), float(os.environ.get("PNP_MAX_VELOCITY_SCALING", "1.0")))
        req.max_velocity_scaling_factor = req.max_acceleration_scaling_factor = scaling
        req.goal_constraints = [constraints]
        g = MoveGroup.Goal()
        g.request = req
        g.planning_options.plan_only = plan_only
        g.planning_options.replan = False
        return g

    def _run(self, goal, timeout=120.0) -> int:
        return self._run_full(goal, timeout)[0]

    def unstick(self, inside=None) -> bool:
        """A joint that settled a hair PAST its planning limit (physics overshoot after a move that ended at the
        limit, e.g. joint4 0.4 mrad past) makes MoveIt reject every plan (START_STATE_INVALID, "Start state out of
        bounds"; its fix_start_state option doesn't cover it). Nudge such joints back `inside` the limit with a
        tiny direct controller move. -> True if something was moved.
        The REAL SO-101 (PNP_UNSTICK_INSIDE, ~3 deg): its folded rest lies past the elbow limit, and gravity pulls a
        nudged joint back ~1.5 deg (servo compliance) -- 2 mrad inside wasn't enough, every plan stayed invalid."""
        a = self.cfg["arm"]
        inside = float(os.environ.get("PNP_UNSTICK_INSIDE", "0.002")) if inside is None else inside
        q = self.arm_q()
        lo = np.array([a["position_limits"][j][0] for j in a["joints"]]) + a["limit_margin"]
        hi = np.array([a["position_limits"][j][1] for j in a["joints"]]) - a["limit_margin"]
        past = float(np.max(np.maximum(lo - q, q - hi)))
        # how far past still counts as "resting there" (PNP_UNSTICK_MAX: the real arm's folded rest reads 0.18 rad
        # past shoulder_lift); beyond that something is wrong: leave it
        if past <= 0 or past > float(os.environ.get("PNP_UNSTICK_MAX", "0.05")):
            return False
        q_ok = np.clip(q, lo + inside, hi - inside)
        t = max(0.5, float(np.max(np.abs(q_ok - q))) / 0.08)              # slow: <= ~4.6 deg/s
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = list(a["joints"])
        g.trajectory.points = [JointTrajectoryPoint(positions=[float(v) for v in q_ok],
                                                    time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9)))]
        self._action(self.traj, g, 10.0 + t)
        time.sleep(0.3)
        moved = [f"{j} {1000 * (b - c):+.1f} mrad" for j, b, c in zip(a["joints"], q_ok, q) if abs(b - c) > 1e-9]
        self.node.get_logger().warn(f"start state was past a joint limit: nudged {', '.join(moved)}")
        return True

    def _run_full(self, goal, timeout=120.0):
        """-> (error code, final arm joint positions of the planned trajectory or None)"""
        pausable = not goal.planning_options.plan_only
        while True:
            try:
                res = self._action(self.move, goal, timeout, pausable)
                break
            except Paused:
                continue                     # MoveIt plans the same goal again from where the arm stopped
        if res.error_code.val == MoveItErrorCodes.START_STATE_INVALID and                 not goal.request.start_state.joint_state.name and self.unstick():   # (from the CURRENT state)
            res = self._action(self.move, goal, timeout)                  # once more, from inside the limits
        pts, names = res.planned_trajectory.joint_trajectory.points, res.planned_trajectory.joint_trajectory.joint_names
        q_end = None
        idx = [list(names).index(j) for j in self.cfg["arm"]["joints"]] if all(j in names for j in self.cfg["arm"]["joints"]) \
            else None
        # the planned path's arm joint waypoints (a preview shows them: /display_planned_path)
        self.last_path = [[p.positions[i] for i in idx] for p in pts] if pts and idx else []
        if pts:
            last = dict(zip(names, pts[-1].positions))
            q_end = [last[j] for j in self.cfg["arm"]["joints"]] if all(j in last for j in self.cfg["arm"]["joints"]) else None
        return int(res.error_code.val), q_end

    def move_joints(self, q, scaling=None, ptp=False) -> int:
        """Arm to joint values: OMPL (free-space), or ptp=True: Pilz PTP, a straight line in joint space (short,
        predictable, still collision-checked)."""
        c = Constraints()
        c.joint_constraints = [JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005,
                                               tolerance_below=0.005, weight=1.0)
                               for j, v in zip(self.cfg["arm"]["joints"], q)]
        pipeline, planner = ("pilz_industrial_motion_planner", "PTP") if ptp else ("ompl", "")
        return self._run(self._request(c, pipeline, planner, scaling or self.free_scaling))

    def move_pose(self, xyz, quat_xyzw, linear=False, scaling=None, plan_only=False, ptp=False) -> int:
        """TCP to a pose. linear=True: Pilz LIN (straight line); ptp=True: Pilz PTP (a straight line in JOINT space
        to the IK solution nearest the start: short and predictable); else OMPL (collision-aware free-space).
        plan_only=True: only check that a plan exists (nothing moves)."""
        return self.plan_pose(xyz, quat_xyzw, linear, scaling, plan_only, ptp=ptp)[0]

    def plan_pose(self, xyz, quat_xyzw, linear=False, scaling=None, plan_only=True, start_q=None, ptp=False):
        """Like move_pose, but returns (code, final joint positions of the plan). With start_q: plan as if the arm
        were at start_q (plan_only), e.g. "could I go straight down from where that other plan ends?"."""
        if self.chain is not None:
            return self._plan_pose_numeric(xyz, quat_xyzw, linear, scaling, plan_only, start_q)
        c = Constraints()
        pc = PositionConstraint(link_name=self.tcp, weight=1.0)
        pc.header.frame_id = self.base
        vol = BoundingVolume()
        vol.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.001])]
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, xyz)
        p.orientation.w = 1.0
        vol.primitive_poses = [p]
        pc.constraint_region = vol
        q = Pose().orientation
        q.x, q.y, q.z, q.w = map(float, quat_xyzw)
        oc = OrientationConstraint(link_name=self.tcp, orientation=q, weight=1.0, absolute_x_axis_tolerance=0.01,
                                   absolute_y_axis_tolerance=0.01, absolute_z_axis_tolerance=0.01)
        oc.header.frame_id = self.base
        c.position_constraints, c.orientation_constraints = [pc], [oc]
        pipeline, planner = (("pilz_industrial_motion_planner", "LIN") if linear else
                             ("pilz_industrial_motion_planner", "PTP") if ptp else ("ompl", ""))
        scaling = scaling or self.free_scaling
        code, q_end = self._run_full(self._request(c, pipeline, planner, scaling, plan_only, start_q))
        # Pilz LIN keeps the TCP on a straight line at the requested Cartesian speed; in some arm configurations
        # that needs more joint acceleration than the limits allow, and Pilz refuses ("Joint acceleration limit
        # ... violated. Set the acceleration scaling factor lower!"). A failed plan never moves the robot, so
        # just retry slower: fast where the geometry allows, automatically gentler where it doesn't.
        # (plan_only requests report this as the generic FAILURE instead of PLANNING_FAILED)
        while linear and code in (MoveItErrorCodes.PLANNING_FAILED, MoveItErrorCodes.FAILURE) and scaling > 0.1:
            scaling /= 2
            self.node.get_logger().warn(f"LIN plan failed, retrying at scaling {scaling:.2f}")
            code, q_end = self._run_full(self._request(c, pipeline, planner, scaling, plan_only, start_q))
        return code, q_end

    # --- pose goals by our own IK (arm.ik: numeric) -------------------------------------------------------
    def ik(self, xyz, quat_xyzw, seed=None):
        """Joint values for a TCP pose, or None. Seeded from `seed` / the current joints (keeps the arm branch)."""
        q, pe, ae = self.chain.ik(xyz, target_R=quat_to_matrix(quat_xyzw),
                                  q0=self.arm_q() if seed is None else np.asarray(seed, float), seeds=8)
        ok = q is not None and pe < 0.002 and ae < 1.5
        if not ok:
            self.node.get_logger().info(f"IK: no solution for {np.round(xyz, 3).tolist()} "
                                        f"(best {pe * 1000:.1f} mm / {ae:.1f} deg off)")
        return q if ok else None

    def _plan_pose_numeric(self, xyz, quat_xyzw, linear, scaling, plan_only, start_q):
        """Pose goal = IK -> joint goal. linear=True: a straight TCP line as a sequence of IK solutions, executed
        directly by the arm controller (short approach / lift moves near the object; the target region is free).
        Otherwise MoveIt plans a collision-free path to the joint goal (OMPL)."""
        start = self.arm_q() if start_q is None else np.asarray(start_q, float)
        q_goal = self.ik(xyz, quat_xyzw, seed=start)
        if q_goal is None:
            return MoveItErrorCodes.NO_IK_SOLUTION, None
        if not linear:
            c = Constraints()
            c.joint_constraints = [JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005,
                                                   tolerance_below=0.005, weight=1.0)
                                   for j, v in zip(self.cfg["arm"]["joints"], q_goal)]
            return self._run_full(self._request(c, "ompl", "", scaling or self.free_scaling, plan_only, start_q))
        # straight line: IK along the segment from the CURRENT tcp pose (start_q: from FK of start_q)
        p0 = self.chain.fk(start)[:3, 3]
        n = max(2, int(np.ceil(np.linalg.norm(np.asarray(xyz) - p0) / 0.005)) + 1)       # every 5 mm
        qs, q = [start], start
        for t in np.linspace(0, 1, n)[1:]:
            q = self.ik(p0 + t * (np.asarray(xyz) - p0), quat_xyzw, seed=q)
            if q is None or np.max(np.abs(q - qs[-1])) > 0.5:       # no solution / a branch flip mid-line
                return MoveItErrorCodes.NO_IK_SOLUTION, None
            qs.append(q)
        # these lines skip MoveIt's planner: check every waypoint against the planning scene ourselves (obstacles,
        # Octomap, the held object), like the planner would (the FR3's Pilz LIN moves are checked by MoveIt)
        self.last_path = [list(map(float, q)) for q in qs]
        bad = self.first_invalid(qs[1:])
        if bad is not None:
            self.node.get_logger().warn(f"straight line blocked at waypoint {bad + 1}/{len(qs) - 1}: {self._last_contacts}")
            return MoveItErrorCodes.GOAL_IN_COLLISION, None
        if plan_only:
            return MoveItErrorCodes.SUCCESS, list(qs[-1])
        v = float(self.cfg["arm"].get("max_velocity", 1.0)) * (scaling or self.free_scaling)
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = list(self.cfg["arm"]["joints"])
        t = 0.0
        for a, b in zip(qs[:-1], qs[1:]):
            t += max(float(np.max(np.abs(b - a))) / v, 0.03)
            g.trajectory.points.append(JointTrajectoryPoint(
                positions=[float(x) for x in b], time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))))
        while True:
            try:
                res = self._action(self.traj, g, 30.0 + t * 3, pausable=True)
                break
            except Paused:
                # resume the line from the waypoint nearest to where the arm stopped, re-timed from there
                q_now = self.arm_q()
                k = int(np.argmin([np.max(np.abs(np.asarray(q) - q_now)) for q in qs]))
                g = FollowJointTrajectory.Goal()
                g.trajectory.joint_names = list(self.cfg["arm"]["joints"])
                t, prev = 0.0, q_now
                for q in qs[k:]:
                    t += max(float(np.max(np.abs(np.asarray(q) - prev))) / v, 0.03)
                    g.trajectory.points.append(JointTrajectoryPoint(
                        positions=[float(x) for x in q], time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))))
                    prev = np.asarray(q)
        return (MoveItErrorCodes.SUCCESS if res.error_code == 0 else MoveItErrorCodes.CONTROL_FAILED), list(qs[-1])

    def first_invalid(self, qs):
        """Index of the first arm configuration in qs that MoveIt's planning scene says is in collision, or None."""
        if not self.validity.wait_for_service(timeout_sec=2.0):
            return None                                   # no move_group yet: nothing to check against
        self._last_contacts = ""
        for i, q in enumerate(qs):
            req = GetStateValidity.Request()
            req.group_name = self.cfg["arm"]["planning_group"]
            req.robot_state.is_diff = True
            req.robot_state.joint_state.name = list(self.cfg["arm"]["joints"])
            req.robot_state.joint_state.position = [float(v) for v in q]
            res = self.wait(self.validity.call_async(req), 2.0)
            if res is not None and not res.valid:
                self._last_contacts = ", ".join(sorted({f"{c.contact_body_1}-{c.contact_body_2}" for c in res.contacts}))
                return i
        return None

    def gripper(self, width: float, max_effort: float, timeout=6.0):
        """Open (or close) the gripper to an opening WIDTH [m] (converted to the joint by the gripper model) and wait
        until the controller reports it done OR the jaws stop moving (blocked by something: the controller's stall
        detection is unreliable in the sim, and a goal that never finishes must not hang the task).
        -> the width reached."""
        if not self.grip.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("gripper action server not available")
        g = GripperCommand.Goal()
        g.command.position, g.command.max_effort = self.gripper_model.command(width), float(max_effort)
        handle = self.wait(self.grip.send_goal_async(g), 10.0)
        if handle is None or not handle.accepted:
            raise RuntimeError("gripper goal rejected")
        res = handle.get_result_async()
        start, last_w, last_change = time.time(), self.gripper_width(), time.time()
        while not res.done() and time.time() - start < timeout:
            time.sleep(0.05)
            w = self.gripper_width()
            if abs(w - last_w) > 0.0005:
                last_w, last_change = w, time.time()
            elif time.time() - last_change > 0.5 and time.time() - start > 0.4:
                break                                     # settled (reached or blocked)
        if abs(self.gripper_width() - width) > 0.006:
            self.node.get_logger().warn(f"gripper stopped at {self.gripper_width() * 1000:.1f} mm "
                                        f"(asked {width * 1000:.1f} mm): blocked?")
        return self.gripper_width()

    def close_on_object(self, max_effort: float, settle_s=0.25, timeout=4.0) -> float:
        """Close the fingers and return the width once they have STOPPED MOVING.

        Why not just wait for the GripperCommand result: when the fingers stop on an object, the controller
        reports success only after its "stall" detection triggers (finger speed below a threshold for a
        while); in the sim the squeezing fingers jitter enough that this is unreliable (seen in the benchmark:
        the close sometimes never finished). So we send the goal and watch the width ourselves. The goal stays
        ACTIVE, so the controller keeps squeezing; the next gripper command (open) simply replaces it."""
        if not self.grip.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("gripper action server not available")
        g = GripperCommand.Goal()
        g.command.position, g.command.max_effort = self.gripper_model.closed_command(), float(max_effort)
        handle = self.wait(self.grip.send_goal_async(g), 10.0)
        if handle is None or not handle.accepted:
            raise RuntimeError("gripper close goal rejected")
        start, last_w, last_change = time.time(), self.gripper_width(), time.time()
        while time.time() - start < timeout:
            time.sleep(0.05)
            w = self.gripper_width()
            if abs(w - last_w) > 0.0005:            # still moving (> 0.5 mm since the last change)
                last_w, last_change = w, time.time()
            elif time.time() - last_change > settle_s and time.time() - start > 0.3:
                break
        return self.gripper_width()

    # --- planning scene ----------------------------------------------------------------------------------
    def _apply(self, scene: PlanningScene, strict=True) -> bool:
        if not self.scene.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/apply_planning_scene not available")
        res = self.wait(self.scene.call_async(ApplyPlanningScene.Request(scene=scene)), 10.0)
        ok = res is not None and res.success
        if not ok and strict:
            raise RuntimeError("apply_planning_scene failed")
        return ok

    def add_box(self, name, center, size, yaw=0.0):
        """Add (or replace: same id) a box in the base frame, rotated `yaw` about z."""
        co = CollisionObject(id=name, operation=CollisionObject.ADD)
        co.header.frame_id = self.base
        co.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(s) for s in size])]
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, center)
        p.orientation.z, p.orientation.w = float(np.sin(yaw / 2)), float(np.cos(yaw / 2))
        co.primitive_poses = [p]
        s = PlanningScene(is_diff=True)
        s.world.collision_objects = [co]
        self._apply(s)

    def remove_objects(self, names):
        if not names:
            return
        s = PlanningScene(is_diff=True)
        s.world.collision_objects = [CollisionObject(id=n, operation=CollisionObject.REMOVE) for n in names]
        self._apply(s, strict=False)

    @staticmethod
    def _object_shape(obj_cfg, T, pad=0.0):
        """An object's catalog shape as a MoveIt primitive at pose T (its own frame), grown by `pad` on every side.
        -> (SolidPrimitive, the primitive's pose)."""
        if obj_cfg["shape"] == "cylinder":
            # MoveIt's CYLINDER axis is the primitive's z: turn z onto the object's up axis
            prim = SolidPrimitive(type=SolidPrimitive.CYLINDER,
                                  dimensions=[float(obj_cfg["dims_m"][2]) + 2 * pad, float(obj_cfg["dims_m"][0] / 2) + pad])
            return prim, T @ make_transform(rotation_between([0, 0, 1], up_axis(obj_cfg)), [0, 0, 0])
        return SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(v) + 2 * pad for v in object_extents(obj_cfg)]), T

    @staticmethod
    def _pose_msg(T):
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, matrix_to_quat(T[:3, :3]))
        return p

    def add_object(self, name, obj_cfg, T_base_obj, pad=0.0):
        """An object (catalog shape, grown by `pad`) as a WORLD collision object at its pose in the base frame. The pick
        target is one: MoveIt then keeps its voxels out of the Octomap (its own shape filter, like the table's) -- and
        nothing else's, so an obstacle right beside it stays whole (a cut-out around the target hid one)."""
        prim, T = self._object_shape(obj_cfg, np.asarray(T_base_obj, float), pad)
        co = CollisionObject(id=name, operation=CollisionObject.ADD)
        co.header.frame_id = self.base
        co.primitives, co.primitive_poses = [prim], [self._pose_msg(T)]
        s = PlanningScene(is_diff=True)
        s.world.collision_objects = [co]
        self._apply(s)

    def allow_contact(self, name, links, allowed=True) -> bool:
        """Allow (or forbid again) contact between the collision object `name` and the robot `links` in MoveIt's
        allowed collision matrix: the open jaws must reach around the pick target, nothing else may touch it.
        A matrix in a scene diff REPLACES the whole matrix, so: read it, change these entries, write it back."""
        if not self.get_scene.wait_for_service(timeout_sec=5.0):
            return False
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        res = self.wait(self.get_scene.call_async(req), 5.0)
        if res is None:
            return False
        acm = res.scene.allowed_collision_matrix
        names = list(acm.entry_names)
        for n in [name, *links]:
            if n not in names:
                names.append(n)
                for row in acm.entry_values:
                    row.enabled.append(False)
                acm.entry_values.append(AllowedCollisionEntry(enabled=[False] * len(names)))
        acm.entry_names = names
        i = names.index(name)
        for link in links:
            j = names.index(link)
            acm.entry_values[i].enabled[j] = acm.entry_values[j].enabled[i] = bool(allowed)
        s = PlanningScene(is_diff=True)
        s.allowed_collision_matrix = acm
        return self._apply(s, strict=False)

    def attach_object(self, name, obj_cfg, T_tcp_obj):
        """Attach the grasped object to the TCP (for collision checking during transport only; the physics
        grasp in the sim is real), as its catalog shape at the measured grasp offset T_tcp_obj."""
        prim, T = self._object_shape(obj_cfg, np.asarray(T_tcp_obj, float))
        aco = AttachedCollisionObject(link_name=self.tcp)
        aco.object.id, aco.object.operation = name, CollisionObject.ADD
        aco.object.header.frame_id = self.tcp
        aco.object.primitives = [prim]
        p = Pose()
        p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
        q = matrix_to_quat(T[:3, :3])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
        aco.object.primitive_poses = [p]
        aco.touch_links = list(self.cfg.get("hand_links", [self.tcp]))
        s = PlanningScene(is_diff=True)
        s.robot_state.is_diff = True
        s.robot_state.attached_collision_objects = [aco]
        self._apply(s)

    def detach(self, name) -> bool:
        """Remove the attached object (and any world copy of it). Best effort: MoveIt rejects removing
        something that isn't there, which is fine here (e.g. the clean-up at the start of a task)."""
        aco = AttachedCollisionObject(link_name=self.tcp)
        aco.object.id, aco.object.operation = name, CollisionObject.REMOVE
        s = PlanningScene(is_diff=True)
        s.robot_state.is_diff = True
        s.robot_state.attached_collision_objects = [aco]
        ok = self._apply(s, strict=False)
        w = PlanningScene(is_diff=True)
        w.world.collision_objects = [CollisionObject(id=name, operation=CollisionObject.REMOVE)]
        self._apply(w, strict=False)                         # MoveIt turns a detached object into a world object
        return ok

    def clear_octomap(self, refill_s=0.6) -> bool:
        """Forget everything the depth camera told MoveIt (Layer B) and give it `refill_s` to look again."""
        if not self.clear_octomap_cli.wait_for_service(timeout_sec=2.0):
            return False
        ok = self.wait(self.clear_octomap_cli.call_async(Empty.Request()), 5.0) is not None
        time.sleep(refill_s)
        return ok

    # --- perception --------------------------------------------------------------------------------------
    def detect(self, object_class, timeout=3.0):
        """-> list of (center_xyz np.array, score, yaw), FRESH (frame captured after this call). yaw: a box's long
        side, (-pi/2, pi/2] (the detector's minimum-area rectangle); 0 for cylinders."""
        if not self.detect_cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/perception/detect not available (is the detector running?)")
        res = self.wait(self.detect_cli.call_async(DetectObjects.Request(object_class=object_class, timeout_s=timeout)),
                        timeout + 5.0)
        if res is None or not res.success:
            return None
        out = []
        for d in res.detections.detections:
            o = d.bbox.center.orientation
            out.append((np.array([d.bbox.center.position.x, d.bbox.center.position.y, d.bbox.center.position.z]),
                        d.results[0].hypothesis.score, float(2 * np.arctan2(o.z, o.w))))   # pure yaw quaternion
        return out


def error_name(code: int) -> str:
    return ERROR_NAMES.get(code, str(code))
