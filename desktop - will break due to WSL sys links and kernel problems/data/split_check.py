"""Check the robot / brain split over the zenoh link (run on the BRAIN side, i.e. the PC container):
the Jetson's state arrives, its allow-listed services / actions work, the rest is NOT reachable, and MoveIt on the
Jetson moves the arm when the PC asks.   bash /repo/data/rospy.sh /repo/data/split_check.py
"""
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import SetBool, Trigger

from pnp_core.catalog import load_robot
from pnp_manipulation.robot_interface import RobotInterface

ok_all = True


def wait(f, timeout):
    """A future's result, or None after `timeout` s (never hang: a call that can't cross the link just fails)."""
    end = time.time() + timeout
    while not f.done() and time.time() < end:
        time.sleep(0.05)
    return f.result() if f.done() else None


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""), flush=True)


rclpy.init()
n = rclpy.create_node("split_check")
ex = MultiThreadedExecutor(num_threads=4)
ex.add_node(n)
threading.Thread(target=ex.spin, daemon=True).start()
cfg = load_robot("so101")
srv = subprocess.run(["ros2", "service", "list"], capture_output=True, text=True).stdout.split()
act = subprocess.run(["ros2", "action", "list"], capture_output=True, text=True).stdout.split()
for s in ("/so101_bridge/enable", "/wrist_map/scan", "/apply_planning_scene", "/check_state_validity"):
    check(f"service {s} visible from the PC", s in srv)
for a in ("/move_action", "/so101_arm_controller/follow_joint_trajectory", "/so101_gripper/gripper_cmd"):
    check(f"action {a} visible from the PC", a in act)
for s in ("/controller_manager/switch_controller", "/controller_manager/unload_controller", "/so101_bridge/set_parameters"):
    check(f"service {s} NOT reachable from the PC (allow-list)", s not in srv)

R = RobotInterface(n, cfg)
check("robot ready (joint states + move_group over the link)", R.wait_ready(timeout=20.0))
en = n.create_client(SetBool, "/so101_bridge/enable")
en.wait_for_service(timeout_sec=5.0)
r = wait(en.call_async(SetBool.Request(data=True)), 10.0)
# 2026-10-11 in the PC emulation: FAILS -- the zenoh bridge drops Fast DDS service requests ("received invalid
# request ... less than 20 bytes"); topics are fine. See docs/learning/22_jetson_d405.md, "Services over the link".
check("torque on via /so101_bridge/enable", r is not None and r.success, r.message if r else "no answer in 10 s")
time.sleep(1.0)
home = np.array(cfg["arm"]["home"], float)
goal = home + np.array([0.3, -0.2, 0.2, 0.1, 0.3])
t0 = time.time()
code = R.move_joints(goal)
err = float(np.abs(R.arm_q() - goal).max())
check(f"MoveIt on the Jetson moved the arm ({time.time() - t0:.1f} s)", code == 1 and err < 0.02,
      f"code {code}, max joint error {np.degrees(err):.2f} deg")
code = R.move_joints(home)
check("and back home", code == 1, f"code {code}")
sc = n.create_client(Trigger, "/wrist_map/scan")
sc.wait_for_service(timeout_sec=5.0)
r = wait(sc.call_async(Trigger.Request()), 40.0)
check("wrist scan answers over the link (no camera in the emulator: a clean refusal)", r is not None,
      r.message if r else "no answer")
wait(en.call_async(SetBool.Request(data=False)), 5.0)
print("ALL PASS" if ok_all else "SOME FAILED")
ex.shutdown()
rclpy.try_shutdown()
