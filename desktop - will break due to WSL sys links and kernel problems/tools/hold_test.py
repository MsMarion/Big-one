"""How close do the arm joints get to their goals? (the so101_bridge hold loop, pnp_core/servo_hold.py) Moves to a few
poses and prints goal - measured per joint over 3 s. Torque must be on. On the Jetson:
    docker exec jetson-robot-1 bash -ic "python3 /repo/tools/hold_test.py"
2026-10-11, real SO-101, hold_ki 2.0: settled 0.1-0.4 deg everywhere (before: 1.5-3.5 deg on the elbow, stretched out).
"""
import sys, time, threading; sys.path.insert(0, "/repo")
import numpy as np, rclpy
from rclpy.executors import MultiThreadedExecutor
from pnp_core.catalog import load_robot
from pnp_manipulation.robot_interface import RobotInterface
rclpy.init(); n = rclpy.create_node("hold_test"); ex = MultiThreadedExecutor(4); ex.add_node(n); threading.Thread(target=ex.spin, daemon=True).start()
cfg = load_robot(); R = RobotInterface(n, cfg); R.wait_ready(20)
for label, deg in (("look pose (stretched out)", [0, 0, 0, 80, 0]), ("hover-like pose", [-10, -20, 30, 75, 0]), ("look pose again", [0, 0, 0, 80, 0])):
    goal = np.radians(deg)
    code = R.move_joints(goal, scaling=0.15, ptp=True)
    t0 = time.time(); rows = []
    for t in (0.0, 0.5, 1.0, 2.0, 3.0):
        while time.time() - t0 < t: time.sleep(0.02)
        e = np.degrees(R.arm_q() - goal); rows.append(f"{t:.1f}s: max {np.abs(e).max():.2f} deg " + str(np.round(e, 2).tolist()))
    print(f"== {label}: move code {code}"); print("   " + "\n   ".join(rows))
ex.shutdown(); rclpy.try_shutdown()
