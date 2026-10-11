"""First motions of a robot (sim, mock or REAL): home, then the corners of the scene's pick area from above, the gripper.

    python3 /repo/tools/motion_check.py                    # the active scene's robot, dev speed
    python3 /repo/tools/motion_check.py --scaling 0.1      # a real arm the first time: 10 % speed, e-stop in reach

Every move goes through the normal stack (MoveIt, our IK for <6-joint arms, ros2_control), so this checks the whole
chain: description, IK, planning, controllers, hardware interface, gripper. It touches nothing: the TCP stays
`--height` above the table. Each step prints what it commands and where the arm ended up (TF), and stops at the
first failure. On a real robot run it first at low speed, watching the arm.
"""

import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from pnp_core.catalog import load_robot, load_scene
from pnp_core.geometry import matrix_to_quat
from pnp_core.object_grasp import top_down_R
from pnp_manipulation.robot_interface import RobotInterface, error_name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scaling", type=float, default=None, help="velocity scaling (default: the robot's dev profile)")
    ap.add_argument("--height", type=float, default=0.06, help="TCP height above the table [m]")
    a = ap.parse_args()
    rclpy.init()
    cfg, scene = load_robot(), load_scene()
    node = Node("pnp_motion_check")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(node, cfg)
    R.free_scaling = a.scaling or cfg["arm"]["speed_profiles"]["dev"]["free"]
    if not R.wait_ready(20.0):
        print("robot not ready (joint states / move_group)")
        return 1
    table_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
    r = scene["target"]["spawn_region"]
    pts = [(x, y) for x in r["x"] for y in r["y"]]
    steps = [("home", lambda: R.move_joints(cfg["arm"]["home"]))]
    for x, y in pts:
        for yaw in (0.0, np.pi / 2):
            p = [x, y, table_z + a.height]
            steps.append((f"above ({x:.2f}, {y:.2f}) yaw {np.degrees(yaw):.0f}",
                          lambda p=p, yaw=yaw: R.move_pose(p, matrix_to_quat(top_down_R(yaw)))))
    g = cfg["gripper"]
    steps += [("gripper open", lambda: (R.gripper(g["max_width_m"], 5.0), 1)[1]),
              ("gripper to 20 mm", lambda: (R.gripper(0.02, 5.0), 1)[1]),
              ("home", lambda: R.move_joints(cfg["arm"]["home"]))]
    print(f"{cfg['name']}: {len(steps)} steps at velocity scaling {R.free_scaling:.2f}")
    for name, fn in steps:
        t0 = time.time()
        code = fn()
        p, _ = R.tcp_pose()
        ok = code == 1
        print(f"  {'OK ' if ok else 'FAIL'} {name:<28} {time.time() - t0:4.1f} s   tcp at {np.round(p, 3).tolist()}"
              + ("" if ok else f"   ({error_name(code)})"), flush=True)
        if not ok:
            print("stopped at the first failure")
            return 1
    print(f"all {len(steps)} steps OK; gripper width now {R.gripper_width() * 1000:.1f} mm")
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
