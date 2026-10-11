"""Command round trip through the whole robot link: container -> zenoh -> Isaac -> zenoh -> container.

    python3 /repo/tools/link_check.py        # exit 0 = commands move the (sim) robot, 1 = they don't

Why: topics can be "up" in both directions (joint states arrive at 120 Hz) while COMMANDS silently don't reach
Isaac (seen after several stack restarts against the same sim: the bridge route to Isaac goes stale). Then every
motion fails with CONTROL_FAILED and the arm never leaves home. This check moves the gripper a little and
watches the finger width, so test.ps1 / demo.ps1 can restart sim + bridge when it fails.

It watches Isaac's OWN joint feedback (/isaac_joint_states), not /joint_states: when that feedback stops arriving
(a stale route the other way, seen 2026-10-11), topic_based_ros2_control keeps publishing /joint_states by echoing
the COMMANDS, so the gripper "moves" there, everything looks healthy, and TF / MoveIt silently use the commanded
arm pose instead of the real one (3 mm off at the wrist camera from the arm's sag).
"""

import sys
import time
from pathlib import Path

import rclpy
import rclpy.qos
from control_msgs.action import GripperCommand
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pnp_core.catalog import load_robot  # noqa: E402

ROBOT = load_robot()
FEEDBACK = "/isaac_joint_states"     # the sim's real joint state (ros2_control's input: urdf/fr3_sim.urdf.xacro)


def main(timeout=15.0) -> int:
    rclpy.init()
    node = rclpy.create_node("pnp_link_check")
    js = {}
    node.create_subscription(JointState, FEEDBACK, lambda m: js.update(zip(m.name, m.position)),
                             rclpy.qos.qos_profile_sensor_data)
    grip = ActionClient(node, GripperCommand, ROBOT["gripper"]["action"])
    finger = ROBOT["gripper"]["driven_joint"]

    def spin_until(cond, t):
        end = time.time() + t
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.05)
            if cond():
                return True
        return False

    ok = spin_until(lambda: finger in js, 5.0) and grip.wait_for_server(timeout_sec=10.0)
    if ok:
        start = js[finger]
        target = 0.03 if start > 0.035 else 0.04                     # move 1 cm per finger, whichever way is free
        goal = GripperCommand.Goal()
        goal.command.position, goal.command.max_effort = target, 20.0
        grip.send_goal_async(goal)
        ok = spin_until(lambda: abs(js[finger] - target) < 0.003, timeout)
        goal.command.position = ROBOT["gripper"]["max_width_m"] / 2      # leave it open
        grip.send_goal_async(goal)
        spin_until(lambda: False, 1.0)
    if finger not in js:
        print(f"link check: FAIL (no joint feedback from the sim on {FEEDBACK})")
    else:
        print(f"link check: {'OK (commands reach the robot, its feedback reaches us)' if ok else 'FAIL (gripper did not move: commands do not reach the robot)'}")
    node.destroy_node()
    rclpy.try_shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
