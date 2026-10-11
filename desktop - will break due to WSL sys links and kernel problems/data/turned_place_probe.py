"""Where can the FR3 put the cracker box down ON ITS SIDE? Plan-only MoveIt queries (nothing moves) for the turned
hand at the place pose and at 5 / 10 / 15 cm above it, over a few goals and object yaws.

    bash /repo/data/rospy.sh /repo/data/turned_place_probe.py [rest]
"""
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot, load_scene, load_yaml  # noqa: E402
from pnp_core.geometry import matrix_to_quat  # noqa: E402
from pnp_core.object_grasp import grasps_from_pose  # noqa: E402
from pnp_core.reorient import place_poses, rest_poses  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface, error_name  # noqa: E402

rest_name = sys.argv[1] if len(sys.argv) > 1 else "side_x+"
rclpy.init()
node = rclpy.create_node("turned_place_probe")
ex = MultiThreadedExecutor(num_threads=4)
ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()
cfg, scene = load_robot("fr3"), load_scene("config/scenes/fr3_workcell.yaml")
obj = load_yaml("config/objects.yaml")["cracker_box"]
R = RobotInterface(node, cfg)
R.wait_ready()
time.sleep(1.0)
rests = rest_poses(obj)
T0 = np.eye(4)
T0[:3, :3], T0[:3, 3] = rests["upright"].R, [0.55, 0.03, rests["upright"].height_m]
g = grasps_from_pose(T0, obj, [0.0], 0.0, 20.0, 0.08)[0][0]
table_z = 0.0
for goal in [(0.45, -0.28), (0.60, -0.15), (0.50, 0.0), (0.40, 0.20), (0.35, -0.10)]:
    for yaw_deg in (0, 45, 90, 135, 180, 225, 270, 315):
        out = place_poses(g.T_tcp_obj, obj, rests[rest_name], goal, table_z, 0.005, [np.radians(yaw_deg)], cfg["gripper"],
                          g.width_m)
        if not out:
            print(f"goal {goal} yaw {yaw_deg:3d}: hand check rejects")
            continue
        _, T, _ = out[0]
        q = matrix_to_quat(T[:3, :3])
        res = []
        for dz in (0.0, 0.05, 0.10, 0.15):
            code, qs = R.plan_pose(T[:3, 3] + [0, 0, dz], q, plan_only=True)
            res.append(f"+{dz * 100:.0f}cm:{'ok' if qs is not None and error_name(code) == 'SUCCESS' else error_name(code)}")
        hand = T[:3, 2]
        print(f"goal {goal} yaw {yaw_deg:3d}: tcp z {T[2, 3] * 1000:.0f} mm, hand points ({hand[0]:+.2f}, {hand[1]:+.2f}, "
              f"{hand[2]:+.2f})  " + "  ".join(res), flush=True)
ex.shutdown()
rclpy.try_shutdown()
