"""Does the SIMULATED robot move like its DESCRIPTION says? (robot conformance test, docs/plans/platform_design.md)

    python3 /repo/tools/robot_sim_match.py            # sim + stack running; moves each joint in turn

After a URDF -> USD import, a joint can come out with a flipped axis or a shifted zero: ROS (TF, MoveIt, our IK) then
believes the hand is somewhere else than it really is, and every grasp is off by exactly that. This moves one joint
at a time from home, then compares Isaac's TRUE link pose (sim command link_pose) with forward kinematics of the
robot description at the reported joint values. Anything above a millimeter / a degree is a mismatch.
"""

import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402

from pnp_core import robot_description as rdesc  # noqa: E402
from pnp_core.calibration import pose_error  # noqa: E402
from pnp_core.catalog import load_robot, repo_root  # noqa: E402
from pnp_core.urdf_kin import Chain  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402


def main():
    rclpy.init()
    cfg = load_robot()
    node = Node("pnp_robot_sim_match")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(node, cfg)
    b = Bench(name="pnp_robot_sim_match_sim")
    if not R.wait_ready():
        print("robot not ready")
        return 1
    link = cfg["frames"]["flange"]
    if "description" in cfg:
        urdf = rdesc.urdf(cfg, repo_root(), "mock")
    else:
        urdf = node.get_parameter_or("robot_description", None)
        print("this check needs a profile robot (description: in its config); the FR3 uses Isaac's own USD")
        return 1
    chain = Chain(urdf, cfg["frames"]["base"], link)
    home = np.array(cfg["arm"]["home"], float)
    worst = 0.0
    print(f"{'move':<22} {'pos err mm':>10} {'rot err deg':>11}")
    tests = [("home", home)] + [(f"{j} +0.4", home + 0.4 * np.eye(len(home))[i]) for i, j in enumerate(cfg["arm"]["joints"])]
    for name, q in tests:
        q = np.clip(q, chain.lower + 0.05, chain.upper - 0.05)
        R.move_joints(q)
        time.sleep(1.0)
        qa = R.arm_q()
        T_fk = chain.fk(qa)
        T_sim = np.array(b.sim.call("link_pose", link=link)["T"])
        e = pose_error(T_fk, T_sim)
        worst = max(worst, e[0] / 1.0 + e[1])
        print(f"{name:<22} {e[0]:10.2f} {e[1]:11.2f}" + ("   <- MISMATCH" if e[0] > 1.0 or e[1] > 1.0 else ""))
    R.move_joints(home)
    print("=> sim matches the description" if worst < 2.0 else "=> MISMATCH: check joint axes / zeros of the import")
    rclpy.try_shutdown()
    return 0 if worst < 2.0 else 1


if __name__ == "__main__":
    sys.exit(main())
