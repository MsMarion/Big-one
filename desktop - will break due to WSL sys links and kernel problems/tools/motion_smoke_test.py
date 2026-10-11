"""Phase 2 gate: does the arm/gripper actually move through every layer of the stack?

Run in the pnp container while Isaac Sim (run_sim.py), the zenoh link and bringup.launch.py are running:
    python3 /repo/tools/motion_smoke_test.py

Tests (each prints PASS/FAIL, exit code 1 on any failure):
  1. controller   FollowJointTrajectory straight to the controller: joint 1 +0.3 rad and back
  2. moveit_joint MoveIt (OMPL) plan + execute to a joint-space goal, then back home
  3. moveit_lin   MoveIt Pilz LIN: TCP straight down 5 cm and back up (the motion used for grasping)
  4. gripper      GripperCommand close (no object -> width ~0) then open (width ~0.08 m)
Each motion is verified against /joint_states (which comes from Isaac Sim through ros2_control).

What this teaches: test each layer separately. If (1) fails, it's the controller/hardware link; if only
(2)/(3) fail, it's MoveIt configuration.
"""

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (BoundingVolume, Constraints, JointConstraint, MotionPlanRequest,
                             OrientationConstraint, PositionConstraint)
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnp_core.catalog import load_robot  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROBOT = load_robot()
ARM = ROBOT["arm"]["joints"]
HOME = ROBOT["arm"]["home"]


class Smoke(Node):
    def __init__(self):
        super().__init__("pnp_motion_smoke_test", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, True)])
        self.js = None
        self.create_subscription(JointState, "/joint_states", self._js, 10)
        self.traj = ActionClient(self, FollowJointTrajectory, f"/{ROBOT['arm']['controller']}/follow_joint_trajectory")
        self.grip = ActionClient(self, GripperCommand, ROBOT["gripper"]["action"])
        self.move = ActionClient(self, MoveGroup, "/move_action")
        self.tf = Buffer()
        TransformListener(self.tf, self)

    def _js(self, msg):
        self.js = dict(zip(msg.name, msg.position))

    # --- helpers ---------------------------------------------------------------------------------
    def spin_until(self, pred, timeout):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if pred():
                return True
        return False

    def call(self, client, goal, timeout=60.0):
        if not client.wait_for_server(timeout_sec=20.0):
            raise RuntimeError(f"action server {client._action_name} not available")
        fut = client.send_goal_async(goal)
        self.spin_until(fut.done, 20.0)
        handle = fut.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("goal rejected")
        res = handle.get_result_async()
        self.spin_until(res.done, timeout)
        if not res.done():
            raise RuntimeError("result timeout")
        return res.result().result

    def arm_q(self):
        return np.array([self.js[j] for j in ARM])

    def settle(self, seconds=1.0):
        self.spin_until(lambda: False, seconds)

    def report(self, name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name:13s} {detail}", flush=True)
        return ok

    # --- tests -----------------------------------------------------------------------------------
    def test_controller(self):
        q0 = self.arm_q()
        ok = True
        for target in (q0 + np.r_[0.3, np.zeros(6)], q0):
            g = FollowJointTrajectory.Goal()
            g.trajectory.joint_names = ARM
            g.trajectory.points = [JointTrajectoryPoint(positions=target.tolist(), time_from_start=Duration(sec=3))]
            res = self.call(self.traj, g)
            self.settle(0.5)
            err = np.abs(self.arm_q() - target).max()
            ok &= res.error_code == 0 and err < 0.02
        return self.report("controller", ok, f"max joint error {np.degrees(err):.2f} deg, error_code {res.error_code}")

    def _plan_execute(self, constraints, pipeline="ompl", planner=""):
        req = MotionPlanRequest()
        req.group_name = ROBOT["arm"]["planning_group"]
        req.pipeline_id, req.planner_id = pipeline, planner
        req.num_planning_attempts, req.allowed_planning_time = 3, 5.0
        req.max_velocity_scaling_factor = ROBOT["arm"]["velocity_scaling"]
        req.max_acceleration_scaling_factor = ROBOT["arm"]["acceleration_scaling"]
        req.goal_constraints = [constraints]
        g = MoveGroup.Goal()
        g.request = req
        g.planning_options.plan_only = False
        return self.call(self.move, g, timeout=90.0).error_code.val

    def _joint_goal(self, q):
        c = Constraints()
        c.joint_constraints = [JointConstraint(joint_name=j, position=float(v), tolerance_above=0.005,
                                               tolerance_below=0.005, weight=1.0) for j, v in zip(ARM, q)]
        return c

    def test_moveit_joint(self):
        target = np.array(HOME) + np.array([0.4, 0.2, 0.0, 0.3, 0.0, -0.2, 0.0])
        codes = [self._plan_execute(self._joint_goal(target))]
        self.settle(0.5)
        err = np.abs(self.arm_q() - target).max()
        codes.append(self._plan_execute(self._joint_goal(HOME)))
        self.settle(0.5)
        err_home = np.abs(self.arm_q() - np.array(HOME)).max()
        ok = all(c == 1 for c in codes) and err < 0.02 and err_home < 0.02
        return self.report("moveit_joint", ok, f"codes {codes}, error at goal {np.degrees(err):.2f} deg, at home {np.degrees(err_home):.2f} deg")

    def tcp_pose(self):
        base, tcp = ROBOT["frames"]["base"], ROBOT["frames"]["tcp"]
        self.spin_until(lambda: self.tf.can_transform(base, tcp, rclpy.time.Time()), 5.0)
        t = self.tf.lookup_transform(base, tcp, rclpy.time.Time()).transform
        return np.array([t.translation.x, t.translation.y, t.translation.z]), t.rotation

    def _pose_goal(self, xyz, quat):
        base, tcp = ROBOT["frames"]["base"], ROBOT["frames"]["tcp"]
        c = Constraints()
        pc = PositionConstraint(link_name=tcp, weight=1.0)
        pc.header.frame_id = base
        box = BoundingVolume()
        box.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE, dimensions=[0.002])]
        p = PoseStamped().pose
        p.position.x, p.position.y, p.position.z = map(float, xyz)
        p.orientation.w = 1.0
        box.primitive_poses = [p]
        pc.constraint_region = box
        oc = OrientationConstraint(link_name=tcp, orientation=quat, weight=1.0,
                                   absolute_x_axis_tolerance=0.01, absolute_y_axis_tolerance=0.01,
                                   absolute_z_axis_tolerance=0.01)
        oc.header.frame_id = base
        c.position_constraints, c.orientation_constraints = [pc], [oc]
        return c

    def test_moveit_lin(self):
        p0, q = self.tcp_pose()
        target = p0 - np.array([0.0, 0.0, 0.05])
        codes = [self._plan_execute(self._pose_goal(target, q), "pilz_industrial_motion_planner", "LIN")]
        self.settle(0.5)
        p1, _ = self.tcp_pose()
        codes.append(self._plan_execute(self._pose_goal(p0, q), "pilz_industrial_motion_planner", "LIN"))
        self.settle(0.5)
        err = np.linalg.norm(p1 - target)
        ok = all(c == 1 for c in codes) and err < 0.005
        return self.report("moveit_lin", ok, f"codes {codes}, TCP error at bottom {err * 1000:.1f} mm")

    def test_gripper(self):
        widths = []
        for pos in (0.0, ROBOT["gripper"]["max_width_m"] / 2):
            g = GripperCommand.Goal()
            g.command.position, g.command.max_effort = pos, 30.0
            self.call(self.grip, g, timeout=20.0)
            self.settle(0.5)
            # each finger travels 0..0.04 m; finger_joint2 mimics finger_joint1 -> width = 2 x joint1
            widths.append(2.0 * self.js[ROBOT["gripper"]["driven_joint"]])
        ok = widths[0] < 0.005 and widths[1] > 0.075
        return self.report("gripper", ok, f"closed width {widths[0] * 1000:.1f} mm, open width {widths[1] * 1000:.1f} mm")


def main():
    rclpy.init()
    n = Smoke()
    if not n.spin_until(lambda: n.js is not None and all(j in n.js for j in ARM), 20.0):
        print("[FAIL] no /joint_states with all arm joints - is bringup.launch.py running?")
        return 1
    results = []
    for test in (n.test_controller, n.test_moveit_joint, n.test_moveit_lin, n.test_gripper):
        try:
            results.append(test())
        except Exception as e:  # keep going so we see every layer's status
            results.append(n.report(test.__name__.replace("test_", ""), False, f"exception: {e}"))
    print(f"RESULT: {'PASS' if all(results) else 'FAIL'}", flush=True)
    n.destroy_node()
    rclpy.shutdown()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
