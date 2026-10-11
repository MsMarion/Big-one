"""Live joint angles in degrees (one refreshing line), for the mirror check (docs/learning/22_jetson_d405.md, step 4).

    docker exec -it jetson-robot-1 bash -ic "python3 /repo/tools/joint_monitor.py"      (on the Jetson)

Torque off, move ONE joint at a time and check the sign against this table (from the URDF, base frame: x forward,
y left, z up; "all 0" = the arm straight forward and horizontal, gripper ~29 cm out and ~23 cm up):

    shoulder_pan   +  the arm swings to the robot's RIGHT (clockwise seen from above)
    shoulder_lift  +  the upper arm tips DOWN / forward
    elbow_flex     +  the forearm folds DOWN
    wrist_flex     +  the gripper tips DOWN
    wrist_roll     +  the gripper turns about the forearm (pick a direction, check it's consistent with RViz)
    gripper        +  the jaw OPENS

A joint going the wrong way: drive_mode 1 for that motor in config/calibration/so101_desk/so101_follower.json.
"""
import sys
import time

import rclpy
from sensor_msgs.msg import JointState

ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    rclpy.init()
    n = rclpy.create_node("joint_monitor")
    last = {}
    n.create_subscription(JointState, "/joint_states", lambda m: last.update(zip(m.name, m.position)), 10)
    print("\n\n".join(__doc__.split("\n\n")[2:4]))          # how to check + the sign table
    print("  ".join(f"{j[:11]:>11s}" for j in ORDER) + "   (deg)")
    try:
        while rclpy.ok():
            rclpy.spin_once(n, timeout_sec=0.1)
            if last:
                sys.stdout.write("\r" + "  ".join(f"{last.get(j, float('nan')) * 57.2958:+11.1f}" for j in ORDER))
                sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
