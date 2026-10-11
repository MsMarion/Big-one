"""Put the held brick down at a goal spot, open, back off, and look where it landed. Run after tools/grab.py, on the
robot side (the Jetson):  docker exec jetson-robot-1 bash -ic "python3 /repo/tools/place.py --xy 0.18 -0.07"

    --xy X Y         the goal for the brick's center, base frame (m)        --yaw DEG   its yaw on the desk (default 0)
    --desk-z Z       the desk height in the arm's frame (default: the height the grab's close look measured, -12.4 mm;
                     the configured -2.4 mm is ~1 cm above the real desk in the arm's own frame)
    --no-look        don't check the result

The held brick sits in the TCP frame where the pick's grasp puts it (pnp_core/object_grasp.py: the nominal grasp,
shifted for the fixed jaw). Its bottom goes to desk + 2 mm. Gravity sag: the final command adds the offset between
the controller's REFERENCE and the measured pose at the pose above the goal (the grab's lesson: commands chained off the
measured, sagged pose land low).
"""
import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from control_msgs.msg import JointTrajectoryControllerState
from rclpy.executors import MultiThreadedExecutor

sys.path.insert(0, "/repo/tools")
from hover_over import call  # noqa: E402
from pnp_core.catalog import load_robot, load_scene, load_yaml  # noqa: E402
from pnp_core.geometry import matrix_to_quat  # noqa: E402
from pnp_core.object_grasp import ObjectGrasp, nominal_grasp_offset, shift_for_fixed_jaw, top_down_R  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402


def sag(R, ref):
    """commanded - measured TCP position [m], from the arm controller's reference vs the joint states."""
    if not ref or R.chain is None:
        return np.zeros(3)
    q_ref = np.array([ref[j] for j in R.cfg["arm"]["joints"]])
    return R.chain.fk(q_ref)[:3, 3] - R.chain.fk(R.arm_q())[:3, 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy", type=float, nargs=2, default=[0.18, -0.07])
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--object", default="lego_2x4")
    ap.add_argument("--desk-z", type=float, default=-0.0124)
    ap.add_argument("--no-look", action="store_true")
    a = ap.parse_args()
    cfg, cat = load_robot(), load_yaml("config/objects.yaml")
    oc = cat[a.object]
    rclpy.init()
    n = rclpy.create_node("place")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    ref = {}

    def on_state(m):
        ref.update(zip(m.joint_names, m.reference.positions))
    n.create_subscription(JointTrajectoryControllerState, f"/{cfg['arm']['controller']}/controller_state", on_state, 10)
    R = RobotInterface(n, cfg)
    if not R.wait_ready(timeout=20.0):
        raise SystemExit("robot not ready")
    w = R.gripper_width()
    if w < 0.006:
        raise SystemExit(f"the gripper is closed on nothing ({w * 1000:.1f} mm): nothing to place")
    print(f"holding (jaw {w * 1000:.1f} mm); placing the {a.object} at ({a.xy[0]:.3f}, {a.xy[1]:.3f}), yaw {a.yaw:.0f} deg")

    # where the TCP goes so the brick's center lands on the goal, its bottom 2 mm above the desk
    T_tcp_obj = shift_for_fixed_jaw([ObjectGrasp("nominal", np.eye(4), nominal_grasp_offset(oc), float(oc["grasp"]["width_m"]))],
                                    R.gripper_model.held_offset_y, R.gripper_model.approach_clearance)[0].T_tcp_obj
    h = float(oc["dims_m"][2])
    obj_z = a.desk_z + 0.002 + h / 2
    best = None
    for yaw in (np.radians(a.yaw), np.radians(a.yaw) + np.pi):          # the brick looks the same turned 180 deg
        Rt = top_down_R(yaw)
        p_tcp = np.array([a.xy[0], a.xy[1], obj_z]) - Rt @ T_tcp_obj[:3, 3]
        above = p_tcp + np.array([0.0, 0.0, 0.04])
        code = R.move_pose(above, matrix_to_quat(Rt), scaling=0.1, ptp=True)
        if code == -4:
            pos, _ = R.tcp_pose()
            code = 1 if np.linalg.norm(pos - above) < 0.015 else code
        print(f"  above the goal (TCP yaw {np.degrees(yaw):.0f} deg): {'ok' if code == 1 else f'MoveIt {code}'}")
        if code == 1:
            best = (Rt, p_tcp, above)
            break
    if best is None:
        print("STOP: can't reach above the goal: try a spot nearer the robot")
        return 1
    Rt, p_tcp, above = best
    q = matrix_to_quat(Rt)
    time.sleep(0.8)
    s = sag(R, ref)
    print(f"sag here: commanded - measured = ({s[0] * 1000:+.1f}, {s[1] * 1000:+.1f}, {s[2] * 1000:+.1f}) mm")
    t = load_scene()["table"]
    R.add_box("table", [t["center"][0], t["center"][1], a.desk_z - 0.004 - t["size"][2] / 2], t["size"])
    code = R.move_pose(p_tcp + s, q, linear=True, scaling=0.05)
    print(f"  down to the desk: {'ok' if code in (1, -4) else f'MoveIt {code}'}")
    if code not in (1, -4):
        print("STOP: holding it where it is")
        return 1
    R.gripper(min(float(cfg["gripper"]["max_width_m"]), w + 0.012), 5.0)
    time.sleep(0.5)
    code = R.move_pose(above + s, q, linear=True, scaling=0.1)
    print(f"released (jaw {R.gripper_width() * 1000:.1f} mm), backed off: {'ok' if code in (1, -4) else f'MoveIt {code}'}")
    if a.no_look:
        return 0
    R.move_joints(np.radians([0, 0, 0, 80, 0]), scaling=0.15, ptp=True)
    time.sleep(1.0)
    print("look: " + call(n, "/wrist_map/look", timeout=60.0).message)
    print(f"(goal was ({a.xy[0]:.3f}, {a.xy[1]:.3f}))")
    ex.shutdown()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
