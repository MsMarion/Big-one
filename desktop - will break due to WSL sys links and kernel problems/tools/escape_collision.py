"""Get the arm out of a state MoveIt calls "in collision" (it then refuses to plan: code -10 START_STATE_IN_COLLISION),
e.g. a calibration view that sagged / an open jaw that now overlaps the shoulder in the model. Checks a straight
joint-space line from where the arm is to TARGET (degrees) in steps with MoveIt's state-validity service; moves
(one slow direct controller move) only if every step after the first few is valid -- i.e. the line LEAVES the
collision and stays clear. Torque must be on.
    python3 /repo/tools/escape_collision.py 20.7 -53.2 32.1 90.4 31.8 [--secs 8] [--dry-run]
"""
import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetStateValidity
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", type=float, nargs=5, help="arm joints, degrees")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--allow-start", type=float, default=0.2, help="fraction of the line allowed in collision")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_robot()
    joints = cfg["arm"]["joints"]
    rclpy.init()
    n = rclpy.create_node("pnp_escape")
    ex = MultiThreadedExecutor()
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    js = {}
    n.create_subscription(JointState, "/joint_states", lambda m: js.update(zip(m.name, m.position)), 10)
    end = time.time() + 5
    while not all(j in js for j in joints) and time.time() < end:
        time.sleep(0.05)
    q0 = np.array([js[j] for j in joints])
    q1 = np.radians(a.target)
    print("from (deg):", np.round(np.degrees(q0), 1).tolist(), "\nto   (deg):", np.round(np.degrees(q1), 1).tolist())
    cli = n.create_client(GetStateValidity, "/check_state_validity")
    cli.wait_for_service(timeout_sec=5)
    bad = []
    for k in range(a.steps + 1):
        q = q0 + (q1 - q0) * k / a.steps
        rs = RobotState()
        rs.joint_state.name = list(joints) + [j for j in js if j not in joints]
        rs.joint_state.position = [float(v) for v in q] + [float(js[j]) for j in js if j not in joints]
        r = cli.call(GetStateValidity.Request(robot_state=rs, group_name=cfg["arm"].get("group", "arm")))
        if not r.valid:
            bad.append((k / a.steps, sorted({(c.contact_body_1, c.contact_body_2) for c in r.contacts})))
    for f, c in bad:
        print(f"  {f:4.0%} of the way: in collision {c}")
    late = [f for f, _ in bad if f > a.allow_start]
    if late:
        print(f"NOT moving: the line is still in collision after {a.allow_start:.0%} (pick another target)")
        return 1
    print(f"the line is clear after {max([f for f, _ in bad], default=0):.0%}" + (" (dry run)" if a.dry_run else ""))
    if a.dry_run:
        return 0
    ac = ActionClient(n, FollowJointTrajectory, f"/{cfg['arm']['controller']}/follow_joint_trajectory")
    ac.wait_for_server(timeout_sec=5)
    g = FollowJointTrajectory.Goal()
    g.trajectory.joint_names = list(joints)
    s = int(a.secs)
    g.trajectory.points = [JointTrajectoryPoint(positions=[float(v) for v in q1],
                                                time_from_start=Duration(sec=s, nanosec=int((a.secs - s) * 1e9)))]
    res = ac.send_goal_async(g)
    while not res.done():
        time.sleep(0.05)
    r = res.result().get_result_async()
    while not r.done():
        time.sleep(0.05)
    time.sleep(0.5)
    q = np.array([js[j] for j in joints])
    print("done; joint errors (deg):", np.round(np.degrees(q - q1), 1).tolist())
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
