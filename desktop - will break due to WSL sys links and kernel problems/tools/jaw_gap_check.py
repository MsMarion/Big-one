"""How far are the VISIBLE fingers / jaws from the object they hold? (sim, any robot: the collision shapes vs the looks)

    python3 /repo/tools/jaw_gap_check.py          (sim + stack running; the active scene's robot and target)
        --save data/jaw_held.png                   + the wrist camera's image while it holds the object

Physics touches COLLISION shapes; we see the render meshes. If the collision shapes are bigger (convex hulls of a
curved jaw, contact offsets), the gripper holds the object across a visible GAP (the SO-101 before its SDF jaws:
1.9 mm); if smaller, the fingers visibly sink INTO it. This grasps the target the way pick & place does, lifts it, and
asks Isaac (sim command visual_gap) for the signed distance from each gripper link's visible surface to the object's
visible shape: + = gap, - = overlap. Both links of a good grasp read ~0 (|x| < 0.5 mm).
"""

import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

sys.path.insert(0, "/repo/tools")
from run_benchmark import Bench  # noqa: E402

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_robot  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402
from pnp_core.grasp import choose_yaw  # noqa: E402
from pnp_core.object_grasp import grasps_from_pose, shift_for_fixed_jaw  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402


def main():
    rclpy.init()
    b = Bench(name="pnp_jaw_gap_check")
    scene, obj = b.scene()
    cfg = load_robot(scene.get("robot", "fr3"))
    node = Node("pnp_jaw_gap_check_robot")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(node, cfg)
    R.wait_ready()
    latest = {}
    if "--save" in sys.argv:
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from pnp_core.wrist import wrist_camera
        node.create_subscription(Image, wrist_camera(cfg)["topics"]["color_image"],
                                 lambda m: latest.__setitem__("img", m), qos_profile_sensor_data)
    links = cfg["sim"].get("gripper_links", ["fr3_leftfinger", "fr3_rightfinger"])
    base = cfg["frames"]["base"]

    def T_of(frame):
        t = R.tf.lookup_transform(base, frame, Time()).transform
        return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                              [t.translation.x, t.translation.y, t.translation.z])

    b.sim.wait_ready(timeout=5.0)
    R.move_joints(cfg["arm"]["home"])
    b.sim.call("clear_obstacles")
    b.sim.call("reset_target")
    time.sleep(1.5)
    T0 = T_of("target")
    g_cfg = obj["grasp"]
    grasps, _ = grasps_from_pose(T0, obj, [choose_yaw(0.0, R.tcp_yaw(), g_cfg.get("symmetric_yaw", False))],
                                 R.tcp_yaw(), 20.0, cfg["gripper"]["max_width_m"])
    grasps = shift_for_fixed_jaw(grasps, R.gripper_model.held_offset_y, R.gripper_model.approach_clearance)
    m = cfg.get("grasp_motion", {})
    plan = grasps[0].plan(cfg["gripper"]["max_width_m"], g_cfg["force_n"],
                          approach_height=float(m.get("approach_height_m", 0.10)),
                          lift_height=float(m.get("lift_height_m", 0.15)))
    gr = cfg["gripper"]
    open_w = min(gr["max_width_m"], g_cfg["width_m"] + 2 * gr.get("open_margin_m", 0.02))
    try:
        R.gripper(open_w, g_cfg["force_n"])
        R.move_pose(plan.pregrasp.position, plan.pregrasp.quat_xyzw)
        R.move_pose(plan.grasp.position, plan.grasp.quat_xyzw, linear=True, scaling=0.3)
        R.close_on_object(g_cfg["force_n"])
        R.move_pose(plan.lift.position, plan.lift.quat_xyzw, linear=True, scaling=0.3)     # held in the air
        time.sleep(1.0)
        T_obj = T_of("target")
        gaps = b.sim.call("visual_gap", links=links, timeout=30.0)["links"]
        for link in links:
            g = gaps.get(link)
            print(f"{link:28s} " + ("not next to the object" if g is None else
                                    f"visible gap to the object: {g['gap_mm']:+.2f} mm  ({g['touching']} surface "
                                    f"points within 0.5 mm)"))
        if "img" in latest:
            import cv2
            from record_pick import to_bgr
            out = sys.argv[sys.argv.index("--save") + 1]
            cv2.imwrite(out if out.startswith("/") else f"/repo/{out}", to_bgr(latest["img"]))
            print(f"saved {out}")
        print(f"width (gripper model) {R.gripper_width() * 1000:.1f} mm, {obj.get('yolo_class', 'object')} "
              f"{plan.expected_width_m * 1000:.1f} mm across the grasp; object "
              f"{'HELD' if T_obj[2, 3] - T0[2, 3] > 0.02 else 'NOT held'} ({(T_obj[2, 3] - T0[2, 3]) * 1000:.1f} mm up)")
    finally:
        R.gripper(open_w, g_cfg["force_n"])
        time.sleep(1.0)
        R.move_joints(cfg["arm"]["home"])
        b.sim.call("reset_target")
        ex.shutdown()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
