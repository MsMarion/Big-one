"""Phase 4: do the FAILURE paths behave? Each case must end with a clear outcome code AND a recovered robot
(arm at home, gripper open, nothing attached), ready for the next task.

    python3 /repo/tools/test_failure_paths.py

  1. no_object     can moved out of view         -> NO_OBJECT
  2. unreachable   goal 1.2 m away (out of reach) -> UNREACHABLE  (after a successful grasp)
  3. blocked_goal  a tall box stands ON the goal   -> UNREACHABLE  (Phase 5: known obstacle; can never moves)
  4. cancel        cancel during the transport    -> CANCELED
  5. changed_stop     map frozen per task, a tower appears on the goal while lifting, policy stop -> SCENE_CHANGED
  6. changed_refresh  same, policy refresh: the map is updated before planning the place -> UNREACHABLE (blocked)
Afterwards one normal task must still succeed (the system wasn't left in a bad state).
"""

import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from pnp_interfaces.action import PickPlace
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnp_core.catalog import load_robot  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from run_benchmark import Bench  # noqa: E402
from sim_client import SimClient  # noqa: E402

from pnp_core.catalog import load_robot as _load_robot  # noqa: E402
BASE = _load_robot()["frames"]["base"]     # the active cell's robot base frame

ROBOT = load_robot()
NAMES = {getattr(PickPlace.Result, k): k for k in dir(PickPlace.Result)
         if k.isupper() and isinstance(getattr(PickPlace.Result, k), int)}


class T(Node):
    def __init__(self):
        super().__init__("pnp_failure_tests")
        self.sim = SimClient(self)
        self.pp = ActionClient(self, PickPlace, "/pick_place")
        self.js = {}
        self.create_subscription(JointState, "/joint_states", lambda m: self.js.update(zip(m.name, m.position)), 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.state = ""
        self.clear_octomap_cli = self.create_client(Empty, "/clear_octomap")
        b = Bench(name="pnp_failure_tests_scene")          # the running stack's scene decides the object
        scene, obj = b.scene()
        b.destroy_node()
        self.label = obj["yolo_class"]
        self.center_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2 + obj["dims_m"][2] / 2

    def clear_octomap(self):
        """Perceived-obstacle mode: forget the voxels of boxes that are gone (the tower on the goal), let the camera
        refill the map. Harmless with known obstacles (the Octomap is empty then)."""
        if self.clear_octomap_cli.wait_for_service(timeout_sec=2.0):
            self.wait(self.clear_octomap_cli.call_async(Empty.Request()), 5.0)
            self.wait(rclpy.task.Future(), 1.5)

    def wait(self, fut, timeout):
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def set_param(self, name, value):
        """Set a /pnp_pick_place string parameter (the scene-change cases switch the map mode for themselves)."""
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
        cli = self.create_client(SetParameters, "/pnp_pick_place/set_parameters")
        cli.wait_for_service(timeout_sec=10.0)
        p = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value))
        self.wait(cli.call_async(SetParameters.Request(parameters=[p])), 5.0)

    def run(self, goal_xy, cancel_at_state=None, timeout=240, sim_at_state=None):
        self.pp.wait_for_server(timeout_sec=20)
        g = PickPlace.Goal(object_class=self.label, use_default_place=False, place_x=goal_xy[0], place_y=goal_xy[1])
        h = self.wait(self.pp.send_goal_async(g, feedback_callback=lambda fb: setattr(self, "state", fb.feedback.state)), 20)
        res_f = h.get_result_async()
        end = time.time() + timeout
        canceled = False
        while not res_f.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if cancel_at_state and not canceled and self.state == cancel_at_state:
                h.cancel_goal_async()
                canceled = True
            if sim_at_state and self.state == sim_at_state[0]:
                self.sim.call(sim_at_state[1])                    # change the scene in the middle of the task
                sim_at_state = None
        return res_f.result().result if res_f.done() else None

    def can_back_in_place(self, start_xy, tol=0.03):
        """Ground truth: the can stands upright within `tol` of where it started (put back, not dropped)."""
        rclpy.spin_once(self, timeout_sec=0.2)
        t = self.tf.lookup_transform(BASE, "target", Time()).transform.translation
        upright = abs(t.z - self.center_z) < 0.01
        return bool(upright and np.hypot(t.x - start_xy[0], t.y - start_xy[1]) < tol)

    def recovered(self):
        rclpy.spin_once(self, timeout_sec=0.2)
        q = np.array([self.js.get(j, np.nan) for j in ROBOT["arm"]["joints"]])
        at_home = np.abs(q - np.array(ROBOT["arm"]["home"])).max() < 0.03
        gripper_open = 2 * self.js.get(ROBOT["gripper"]["driven_joint"], 0) > 0.07
        return bool(at_home and gripper_open)


def main():
    rclpy.init()
    t = T()
    results = []

    def case(name, expect, fn, can_should_stay_at=None):
        res = fn()
        got = NAMES.get(res.outcome, "?") if res else "TIMEOUT"
        rec = t.recovered()
        can_ok = True if can_should_stay_at is None else t.can_back_in_place(can_should_stay_at)
        ok = got == expect and rec and can_ok
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name:12s} expected {expect:12s} got {got:12s} robot recovered={rec}"
              f"  can upright where it was={can_ok}  ({res.message if res else ''})", flush=True)

    t.sim.call("clear_obstacles")
    st = t.sim.call("reset_target")
    t.clear_octomap()                              # e.g. a tower left over from a previous run
    goal, start = st["goal"]["xy"], st["target"]["xy"]
    t.sim.call("target_away")
    case("no_object", "NO_OBJECT", lambda: t.run(goal))
    t.sim.call("reset_target")
    t.wait(rclpy.task.Future(), 1.0)
    # rejected by the plan-only reachability check BEFORE grasping: the can must not even move
    case("unreachable", "UNREACHABLE", lambda: t.run([1.2, -0.6]), can_should_stay_at=start)
    t.sim.call("reset_target")
    t.sim.call("block_goal")
    t.wait(rclpy.task.Future(), 2.0)              # let the pick & place node mirror the box into MoveIt
    case("blocked_goal", "UNREACHABLE", lambda: t.run(goal), can_should_stay_at=start)
    t.sim.call("clear_obstacles")
    t.sim.call("reset_target")
    t.clear_octomap()
    t.wait(rclpy.task.Future(), 1.0)
    # canceled while carrying: recovery must put the can back down where it was picked, upright
    case("cancel", "CANCELED", lambda: t.run(goal, cancel_at_state="MOVE_PREPLACE"), can_should_stay_at=start)
    t.sim.call("reset_target")
    case("normal_after", "SUCCESS", lambda: t.run(goal))
    # scene changes during a FROZEN task: the change monitor must notice the tower that appears on the goal
    t.set_param("map_mode", "freeze_per_task")
    for policy, expect in (("stop", "SCENE_CHANGED"), ("refresh", "UNREACHABLE")):
        t.set_param("change_policy", policy)
        t.sim.call("clear_obstacles")
        t.sim.call("reset_target")
        t.clear_octomap()
        case(f"changed_{policy}", expect, lambda: t.run(goal, sim_at_state=("LIFT", "block_goal")),
             can_should_stay_at=start)
    t.set_param("map_mode", "continuous")
    t.set_param("change_policy", "refresh")
    t.sim.call("clear_obstacles")
    t.sim.call("reset_target")
    t.clear_octomap()
    print(f"RESULT: {'PASS' if all(results) else 'FAIL'}")
    rclpy.try_shutdown()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
