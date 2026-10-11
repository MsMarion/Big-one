"""Does the wrist camera agree with the robot's TF? (sim: compared with Isaac's ground truth)

    python3 /repo/tools/wrist_check.py            (sim + stack running; moves the arm above the object and back)
    python3 /repo/tools/wrist_check.py --save data/wrist_view.png    (+ the wrist color image there)

The wrist camera's pose comes from TF (URDF fixed joint on the hand, pnp_core/wrist.py), its image from Isaac (a
camera prim under the hand, same transform). If the two disagree (e.g. Isaac's hand frame differs from the URDF's),
every wrist measurement is off by that error. This puts the arm at the pre-grasp pose above the target, turns one
wrist depth frame into base-frame points and checks:
  table points        z ~ table top       (height / tilt of the mount)
  object top          z ~ object height   (distance along the view)
  object top center   xy ~ ground truth   (sideways / yaw of the mount)
"""

import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402

from pnp_core.catalog import load_robot  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from pnp_core.object_grasp import top_down_R  # noqa: E402
from pnp_core.perception_math import depth_to_meters  # noqa: E402
from pnp_core.geometry import matrix_to_quat  # noqa: E402
from pnp_core.wrist import wrist_camera  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402


def main():
    rclpy.init()
    robot_cfg = load_robot()
    wcfg = wrist_camera(robot_cfg)
    if wcfg is None:
        print(f"robot {robot_cfg['name']} has no wrist camera (config/robot_{robot_cfg['name']}.yaml wrist_camera)")
        return 1
    node = Node("pnp_wrist_check")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(node, robot_cfg)
    frames = {}
    node.create_subscription(Image, wcfg["topics"]["aligned_depth"], lambda m: frames.__setitem__("depth", m),
                             qos_profile_sensor_data)
    node.create_subscription(CameraInfo, wcfg["topics"]["color_info"], lambda m: frames.__setitem__("info", m),
                             qos_profile_sensor_data)
    node.create_subscription(Image, wcfg["topics"]["color_image"], lambda m: frames.__setitem__("color", m),
                             qos_profile_sensor_data)
    b = Bench(name="pnp_wrist_check_scene")
    scene, obj = b.scene()
    b.spin_for(1.0)
    gt, _ = b.gt_pose("target")
    h = obj["dims_m"][2]
    table_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
    if not R.wait_ready():
        print("robot not ready")
        return 1
    # the pre-grasp pose: TCP the robot's approach height above the object's top (FR3 7 cm, SO-101 4 cm)
    approach = float(robot_cfg.get("grasp_motion", {}).get("approach_height_m", 0.07))
    above = np.array([gt[0], gt[1], table_z + h + approach])
    r_obj = 0.5 * float(np.hypot(obj["dims_m"][0], obj["dims_m"][1]))     # the object's footprint radius
    q = matrix_to_quat(top_down_R(R.tcp_yaw()))
    print(f"moving the TCP above the {scene['target']['object']} at {np.round(above, 3).tolist()} ...", flush=True)
    code = R.move_pose(above, q)
    if code != 1:
        print(f"could not move above the object (MoveIt code {code})")
        return 1
    time.sleep(1.0)
    frames.clear()
    t_end = time.time() + 5
    while ("depth" not in frames or "info" not in frames) and time.time() < t_end:
        time.sleep(0.05)
    if "depth" not in frames:
        print("no wrist depth frames (profile rgb? sim restarted with the wrist camera?)")
        return 1
    m, info = frames["depth"], frames["info"]
    depth = depth_to_meters(np.frombuffer(m.data, np.uint16 if m.encoding == "16UC1" else np.float32)
                            .reshape(m.height, m.width), m.encoding)
    tf = R.tf.lookup_transform(robot_cfg["frames"]["base"], m.header.frame_id, Time.from_msg(m.header.stamp)).transform
    T = make_transform(quat_to_matrix([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]),
                       [tf.translation.x, tf.translation.y, tf.translation.z])
    k = info.k
    vs, us = np.mgrid[0:m.height:2, 0:m.width:2]
    d = depth[::2, ::2]
    ok = (d > 0.03) & (d < 1.0)
    pc = np.column_stack([(us[ok] - k[2]) * d[ok] / k[0], (vs[ok] - k[5]) * d[ok] / k[4], d[ok]])
    pb = pc @ T[:3, :3].T + T[:3, 3]
    r = np.hypot(pb[:, 0] - gt[0], pb[:, 1] - gt[1])
    table = pb[(pb[:, 2] < table_z + min(0.01, h / 2)) & (r > r_obj + 0.02)]
    # the object's top surface only (the gripper's fingers are in view too, above it)
    top = pb[(np.abs(pb[:, 2] - (table_z + h)) < min(0.008, h / 3)) & (r < r_obj + 0.005)]
    print(f"wrist frame '{m.header.frame_id}', {m.width}x{m.height}, camera at {np.round(T[:3, 3], 3).tolist()}")
    print(f"table points: {len(table)}, z median {np.median(table[:, 2]) * 1000:+.1f} mm "
          f"(expect {table_z * 1000:+.1f})")
    if len(top):
        c = top[:, :2].mean(axis=0)
        print(f"object top: {len(top)} points, z median {np.median(top[:, 2]):.4f} (expect {table_z + h:.4f}), "
              f"center {np.round(c, 4).tolist()} vs ground truth {np.round(gt[:2], 4).tolist()} -> "
              f"{np.hypot(*(c - gt[:2])) * 1000:.1f} mm")
    else:
        print("object top not seen by the wrist camera")
    if "--save" in sys.argv and "color" in frames:
        import cv2
        c = frames["color"]
        img = np.frombuffer(c.data, np.uint8).reshape(c.height, c.width, -1)
        out = sys.argv[sys.argv.index("--save") + 1]
        cv2.imwrite(out if out.startswith("/") else f"/repo/{out}",
                    cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR) if c.encoding.lower().startswith("rgb") else img)
        print(f"wrist color image saved: {out}")
    R.move_joints(robot_cfg["arm"]["home"])
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
