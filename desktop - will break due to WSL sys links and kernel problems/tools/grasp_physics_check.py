"""Is the sim's grasp PHYSICAL? (no attachment, no teleport: the object must be held by the jaws' friction alone)

    python3 /repo/tools/grasp_physics_check.py          (sim + stack running; any robot / scene; ~1 min)
        --gif data/grasp_physics.gif     film it (overhead + wrist camera, the step on each frame)
        --trace                          the object in the hand every 0.1 s while it's carried

The pick & place node never moves the object itself, and the sim has no code that attaches it to the hand; what
"attaches" it is only MoveIt's planning scene (collision checking) and the pose estimator's ATTACHED state (the
pose follows the hand). This checks the physics with the sim's ground truth (TF `target`), using the same grasp
poses the pick & place node computes (from the TRUE object pose here):
  1. jaws OPEN at the grasp pose, lift      -> the object must stay on the table (nothing pulls it along)
  2. close                                  -> the jaw must stop at the object's width (blocked by it, not closed)
  3. lift + carry sideways                  -> the object rises with the hand; its pose in the hand only shifts
                                               by contact slip/sag (a joint would keep it exactly)
  4. jaws OPEN in the air                   -> the object must FALL back to the table (nothing holds it but friction)
"""

import sys
import threading
import time

import numpy as np
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
from record_pick import to_bgr  # noqa: E402


def main():
    rclpy.init()
    b = Bench(name="pnp_grasp_physics_check")
    scene, obj = b.scene()
    robot_cfg = load_robot(scene.get("robot", "fr3"))
    node = Node("pnp_grasp_physics_check_robot")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(node, robot_cfg)
    gif = sys.argv[sys.argv.index("--gif") + 1] if "--gif" in sys.argv else None
    latest, frames, step, filming = {}, [], ["start"], [bool(gif)]
    if gif:
        import cv2
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from pnp_core.wrist import wrist_camera
        wcfg = wrist_camera(robot_cfg)
        node.create_subscription(Image, "/camera/camera/color/image_raw", lambda m: latest.__setitem__("top", m),
                                 qos_profile_sensor_data)
        if wcfg:
            node.create_subscription(Image, wcfg["topics"]["color_image"], lambda m: latest.__setitem__("wrist", m),
                                     qos_profile_sensor_data)

        def film():
            while filming[0]:
                if "top" in latest:
                    panels = [cv2.resize(to_bgr(latest["top"]), (480, 360))]
                    if "wrist" in latest:
                        panels.append(cv2.resize(to_bgr(latest["wrist"]), (640, 360)))
                    f = np.hstack(panels)
                    cv2.rectangle(f, (0, 0), (f.shape[1], 30), (0, 0, 0), -1)
                    cv2.putText(f, step[0], (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    frames.append(f)
                time.sleep(0.25)
        threading.Thread(target=film, daemon=True).start()
    if not R.wait_ready():
        print("robot not ready")
        return 1
    base = robot_cfg["frames"]["base"]
    g_cfg = obj["grasp"]
    table_z = scene["table"]["center"][2] + scene["table"]["size"][2] / 2
    rest_z = table_z + obj["dims_m"][2] / 2                   # the object's center height lying on the table

    def T_of(frame):
        t = R.tf.lookup_transform(base, frame, Time()).transform
        return make_transform(quat_to_matrix([t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]),
                              [t.translation.x, t.translation.y, t.translation.z])

    def obj_T():
        time.sleep(0.4)                                       # the TF of the sim's ground truth: fresh
        return T_of("target")

    def in_hand():
        return np.linalg.inv(T_of(robot_cfg["frames"]["tcp"])) @ T_of("target")

    def ang(Ra, Rb):
        return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))

    results = []

    def check(name, ok, detail):
        results.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    b.sim.wait_ready(timeout=5.0)
    R.move_joints(robot_cfg["arm"]["home"])
    b.sim.call("clear_obstacles")
    b.sim.call("reset_target")
    time.sleep(1.5)
    T0 = obj_T()
    print(f"{scene['target']['object']} ({obj['mass_kg'] * 1000:.1f} g) at {np.round(T0[:3, 3], 4).tolist()}", flush=True)
    grasps, why = grasps_from_pose(T0, obj, [choose_yaw(0.0, R.tcp_yaw(), False)], R.tcp_yaw(), 20.0,
                                   robot_cfg["gripper"]["max_width_m"])
    grasps = shift_for_fixed_jaw(grasps, R.gripper_model.held_offset_y, R.gripper_model.approach_clearance)
    if not grasps:
        print(f"no grasp: {why}")
        return 1
    m = robot_cfg.get("grasp_motion", {})
    lift_h = float(m.get("lift_height_m", 0.15))
    plan = grasps[0].plan(robot_cfg["gripper"]["max_width_m"], g_cfg["force_n"],
                          approach_height=float(m.get("approach_height_m", 0.10)), lift_height=lift_h)
    g = robot_cfg["gripper"]
    open_w = float(min(g["max_width_m"], g_cfg["width_m"] + 2 * float(g.get("open_margin_m", 0.02))))
    try:
        R.gripper(open_w, g_cfg["force_n"])
        assert R.move_pose(plan.pregrasp.position, plan.pregrasp.quat_xyzw) == 1, "pre-grasp unreachable"
        assert R.move_pose(plan.grasp.position, plan.grasp.quat_xyzw, linear=True, scaling=0.3) == 1, "approach failed"

        print("1. jaws open at the grasp pose, lift", flush=True)
        step[0] = "1. jaws OPEN at the grasp pose, lift: the brick must stay on the table"
        R.move_pose(plan.lift.position, plan.lift.quat_xyzw, linear=True, scaling=0.3)
        T1 = obj_T()
        dz = T1[2, 3] - T0[2, 3]
        check("not lifted without closing", abs(dz) < 0.002, f"object moved {dz * 1000:+.1f} mm up "
              f"(the hand went up {lift_h * 100:.1f} cm)")
        R.move_pose(plan.grasp.position, plan.grasp.quat_xyzw, linear=True, scaling=0.3)

        print("2. close", flush=True)
        step[0] = "2. close: the jaw must stop at the brick"
        w = R.close_on_object(g_cfg["force_n"])
        closed_w = R.gripper_model.width(R.gripper_model.closed_command())
        check("jaw stopped by the object", abs(w - plan.expected_width_m) < 0.004 and w > closed_w + 0.004,
              f"jaw width {w * 1000:.1f} mm (object {plan.expected_width_m * 1000:.1f} mm, fully closed "
              f"{closed_w * 1000:.1f} mm)")

        print("3. lift and carry", flush=True)
        step[0] = "3. lift + carry 4 cm: friction holds it"
        H0 = in_hand()
        R.move_pose(plan.lift.position, plan.lift.quat_xyzw, linear=True, scaling=0.3)
        T3 = obj_T()
        dz = T3[2, 3] - T0[2, 3]
        check("lifted with the hand", abs(dz - lift_h) < 0.006, f"object rose {dz * 1000:.1f} mm "
              f"(hand {lift_h * 1000:.0f} mm)")
        H1 = in_hand()
        side = plan.lift.position + np.array([0.0, -0.04, 0.0])
        trace, moving = [], [True]

        def sample():                                    # the object in the hand + the jaw, while the hand moves
            t0 = time.time()
            while moving[0]:
                try:
                    H = in_hand()
                    trace.append((time.time() - t0, *((H[:3, 3] - H0[:3, 3]) * 1000), R.gripper_width() * 1000,
                                  *(T_of(robot_cfg["frames"]["tcp"])[:3, 3] * 1000)))
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.05)
        th = threading.Thread(target=sample, daemon=True)
        th.start()
        R.move_pose(side, plan.lift.quat_xyzw, linear=True, scaling=0.5)
        time.sleep(0.5)
        moving[0] = False
        th.join()
        H2 = in_hand()
        if "--trace" in sys.argv:
            print("     t [s]   shift in TCP x / y / z [mm]   jaw [mm]   TCP base y [mm]")
            for t, dx, dy, dz_, jw, _, ty, _ in trace[::2]:
                print(f"     {t:5.2f}   {dx:+6.2f} {dy:+6.2f} {dz_:+6.2f}         {jw:5.1f}     {ty:7.1f}")
        slips = [(np.linalg.norm(H[:3, 3] - H0[:3, 3]) * 1000, ang(H0[:3, :3], H[:3, :3])) for H in (H1, H2)]
        print(f"     object in the hand vs right after closing: after lift {slips[0][0]:.2f} mm / {slips[0][1]:.2f} deg, "
              f"after carrying 4 cm {slips[1][0]:.2f} mm / {slips[1][1]:.2f} deg (friction: small shifts; a joint: 0)")
        d = (H2[:3, 3] - H0[:3, 3]) * 1000
        print(f"     the shift in TCP axes [mm]: x {d[0]:+.2f}  y (closing) {d[1]:+.2f}  z (approach) {d[2]:+.2f}; "
              f"jaw width now {R.gripper_width() * 1000:.1f} mm", flush=True)
        check("carried 4 cm", np.linalg.norm(T_of("target")[:2, 3] - side[:2]) < 0.03, "stays with the hand")

        print("4. jaws open in the air", flush=True)
        step[0] = "4. jaws OPEN in the air: the brick must fall"
        R.gripper(open_w, g_cfg["force_n"])
        time.sleep(1.5)
        T4 = obj_T()
        check("falls when released", abs(T4[2, 3] - rest_z) < 0.01,
              f"object at z {T4[2, 3] * 1000:.1f} mm (lying on the table: {rest_z * 1000:.1f} mm; "
              f"the hand is at {side[2] * 1000:.0f} mm)")
        time.sleep(1.0)
    finally:
        filming[0] = False
        R.move_joints(robot_cfg["arm"]["home"])
        b.sim.call("reset_target")
    if gif and frames:
        from PIL import Image as PILImage
        out = gif if gif.startswith("/") else f"/repo/{gif}"
        imgs = [PILImage.fromarray(f[:, :, ::-1]).resize((f.shape[1] * 3 // 4, f.shape[0] * 3 // 4)) for f in frames]
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=250, loop=0, optimize=True)
        print(f"saved {out} ({len(frames)} frames)")
    n_ok = sum(ok for _, ok in results)
    print(f"RESULT: {n_ok}/{len(results)} checks passed -> grasping is {'PHYSICAL' if n_ok == len(results) else 'SUSPECT'}")
    ex.shutdown()
    rclpy.try_shutdown()
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
