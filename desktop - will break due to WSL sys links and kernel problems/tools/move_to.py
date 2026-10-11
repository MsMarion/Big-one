"""Move the arm to a joint pose through MoveIt, slowly. On the REAL arm: hand on the servo power plug.

    python3 /repo/tools/move_to.py 0 0 0 80 0                 # degrees: pan lift elbow wrist_flex wrist_roll
    python3 /repo/tools/move_to.py --home                     # the robot profile's home pose
    python3 /repo/tools/move_to.py 0 0 0 80 0 --scaling 0.1 --enable
        --enable    switch the servo torque on first (/so101_bridge/enable: the bridge holds the current pose, no jump)
        --off       switch the torque off at the end (the arm goes limp: support it!)
Run where move_group runs (on the Jetson: docker exec jetson-robot-1 bash -ic "python3 /repo/tools/move_to.py ...").
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from std_srvs.srv import SetBool
from trajectory_msgs.msg import JointTrajectoryPoint

from pnp_core.catalog import load_robot
from pnp_manipulation.robot_interface import RobotInterface


def call(node, cli, data, timeout=5.0):
    if not cli.wait_for_service(timeout_sec=timeout):
        return None
    f = cli.call_async(SetBool.Request(data=data))
    end = time.time() + timeout
    while not f.done() and time.time() < end:
        time.sleep(0.05)
    return f.result()


def nudge_into_limits(node, cfg, q, margin=0.04, max_out=0.08):      # MoveIt limits are ~0.02 tighter
    """The real arm's folded rest lies a little past the model's elbow/shoulder limits: MoveIt then refuses to plan
    from there (START_STATE_INVALID). Bring such joints just inside with one slow direct controller move (3 s)."""
    lim = cfg["arm"]["position_limits"]
    q2 = np.array(q, float)
    for i, j in enumerate(cfg["arm"]["joints"]):
        lo, hi = lim[j][0] + margin, lim[j][1] - margin
        if q2[i] < lo - max_out or q2[i] > hi + max_out:
            raise SystemExit(f"{j} is {np.degrees(q2[i]):.1f} deg: far outside its limits, not nudging (move it by hand)")
        q2[i] = min(max(q2[i], lo), hi)
    if np.allclose(q2, q):
        return
    print("nudging into the joint limits (deg):", np.round(np.degrees(q2 - q), 1).tolist())
    ac = ActionClient(node, FollowJointTrajectory, f"/{cfg['arm']['controller']}/follow_joint_trajectory")
    ac.wait_for_server(timeout_sec=5.0)
    g = FollowJointTrajectory.Goal()
    g.trajectory.joint_names = list(cfg["arm"]["joints"])
    g.trajectory.points = [JointTrajectoryPoint(positions=q2.tolist(), time_from_start=Duration(sec=3))]
    f = ac.send_goal_async(g)
    while not f.done():
        time.sleep(0.05)
    r = f.result().get_result_async()
    while not r.done():
        time.sleep(0.05)
    time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("q", type=float, nargs="*", help="arm joints in degrees")
    ap.add_argument("--home", action="store_true")
    ap.add_argument("--scaling", type=float, default=0.1)
    ap.add_argument("--enable", action="store_true")
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--nudge-max-deg", type=float, default=4.6, help="how far past a limit the start may be and still be "
                    "nudged in (a folded arm pushed back by hand: up to ~15)")
    ap.add_argument("--ompl", action="store_true", help="plan with OMPL (default: Pilz PTP, a straight line in joint "
                                                        "space: predictable for hand-supervised moves)")
    a = ap.parse_args()
    cfg = load_robot()
    goal = np.array(cfg["arm"]["home"], float) if a.home else np.radians(a.q)
    if len(goal) != len(cfg["arm"]["joints"]):
        raise SystemExit(f"need {len(cfg['arm']['joints'])} joint values ({' '.join(cfg['arm']['joints'])})")

    rclpy.init()
    n = rclpy.create_node("move_to")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(n, cfg)
    if not R.wait_ready(timeout=20.0):
        raise SystemExit("robot not ready (joint states / move_group)")
    print("from (deg):", np.round(np.degrees(R.arm_q()), 1).tolist())
    print("to   (deg):", np.round(np.degrees(goal), 1).tolist(), f"at {a.scaling:.0%} speed")
    en = n.create_client(SetBool, "/so101_bridge/enable")
    if a.enable:
        r = call(n, en, True)
        print("torque on:", r.message if r else "no answer from /so101_bridge/enable")
        if not (r and r.success):
            raise SystemExit("torque not on: not moving")
        time.sleep(1.0)
    nudge_into_limits(n, cfg, R.arm_q(), max_out=np.radians(a.nudge_max_deg))
    t0 = time.time()
    code = R.move_joints(goal, scaling=a.scaling, ptp=not a.ompl)
    err = np.degrees(np.abs(R.arm_q() - goal))
    print(f"move: code {code} ({'ok' if code == 1 else 'FAILED'}) in {time.time() - t0:.1f} s; "
          f"joint errors (deg): {np.round(err, 1).tolist()}")
    if a.off:
        r = call(n, en, False)
        print("torque off:", r.message if r else "no answer")
    ex.shutdown()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
