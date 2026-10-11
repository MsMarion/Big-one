"""Find a brick with the wrist camera and HOVER over it, gripper open, ready to descend -- the pick without the pick.
Run on the robot side (the Jetson):  docker exec jetson-robot-1 bash -ic "python3 /repo/tools/hover_over.py"

    --object lego_2x4      which catalog object (default: the first one the map identified)
    --above 0.04           TCP height above the GRASP point (default: the robot's approach_height_m, as the pick uses)
    --scan                 the multi-view scan instead of one look from the look pose (wider area, more motion)
    --no-look              reuse the last map (/wrist_map/objects) as it is

Steps: look pose -> /wrist_map/look -> the object's pose (map x, y, yaw; resting on the desk) -> the top-down box grasps
the pick node uses (pnp_core/object_grasp.py: across the narrow side, shifted for the SO-101's fixed jaw) -> open the
gripper -> move the TCP to `above` over the first grasp MoveIt reaches -> hold there (torque stays on).
"""
import argparse
import json
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from pnp_core.catalog import load_robot, load_scene, load_yaml
from pnp_core.geometry import matrix_to_quat
from pnp_core.object_grasp import grasps_from_pose, shift_for_fixed_jaw, upright_pose
from pnp_manipulation.robot_interface import RobotInterface

LOOK_POSE_DEG = [0, 0, 0, 80, 0]


def call(node, name, timeout=60.0):
    cli = node.create_client(Trigger, name)
    if not cli.wait_for_service(timeout_sec=5.0):
        raise SystemExit(f"no {name} (wrist_map_node running?)")
    f = cli.call_async(Trigger.Request())
    end = time.time() + timeout
    while not f.done() and time.time() < end:
        time.sleep(0.05)
    if not f.done():
        raise SystemExit(f"{name}: no answer")
    return f.result()


def add_args(ap):
    ap.add_argument("--object")
    ap.add_argument("--above", type=float)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--no-look", action="store_true")
    ap.add_argument("--scaling", type=float, default=0.15)
    ap.add_argument("--at", type=float, nargs=2, metavar=("X", "Y"),
                    help="hover over this spot (base frame, m) without looking: a position from elsewhere (the D435i, "
                         "tools/fixed_look.py), the close looks refine it; --object names what lies there (default lego_2x4)")
    ap.add_argument("--at-yaw", type=float, default=0.0, help="with --at: the brick's yaw (deg)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    a = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("hover_over")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(n, load_robot())
    hover(n, R, a)
    ex.shutdown()
    rclpy.try_shutdown()


def hover(n, R, a):
    """-> dict(name, obj_cfg, grasp, above, width) once hovering, or None."""
    cfg, scene, cat = R.cfg, load_scene(), load_yaml("config/objects.yaml")
    t = scene["table"]
    table_z = t["center"][2] + t["size"][2] / 2
    above = a.above if a.above is not None else float(cfg.get("grasp_motion", {}).get("approach_height_m", 0.04))

    got = {}
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    n.create_subscription(String, "/wrist_map/objects", lambda m: got.__setitem__("o", json.loads(m.data)), latched)
    if not R.wait_ready(timeout=20.0):
        raise SystemExit("robot not ready")

    if not a.no_look and getattr(a, "at", None) is None:
        if not a.scan:
            code = R.move_joints(np.radians(LOOK_POSE_DEG), scaling=a.scaling, ptp=True)
            print(f"look pose: {'ok' if code == 1 else f'MoveIt {code}'}")
            time.sleep(1.0)
        t0 = time.time()
        r = call(n, "/wrist_map/scan" if a.scan else "/wrist_map/look", timeout=180.0)
        print(("scan: " if a.scan else "look: ") + r.message)
        end = time.time() + 5
        while time.time() < end and got.get("o", {}).get("time", 0) < t0:
            time.sleep(0.1)
    end = time.time() + 4                     # the latched map (--no-look reads the last one) takes a moment
    while "o" not in got and time.time() < end and getattr(a, "at", None) is None:
        time.sleep(0.05)

    def identified():
        out = [o for o in got.get("o", {}).get("objects", []) if o.get("object") in cat]
        return [o for o in out if o["object"] == a.object] if a.object else out
    objs = identified()
    if getattr(a, "at", None) is not None:
        objs = [{"object": a.object or "lego_2x4", "xy": list(a.at), "yaw": float(np.radians(a.at_yaw)), "rest": None}]
        print(f"given: {objs[0]['object']} at ({a.at[0]:.3f}, {a.at[1]:.3f}) m, yaw {a.at_yaw:.0f} deg (no look)")
    elif not objs and not a.scan and not a.no_look:
        # one look misses a brick right under the jaws (the jaw strip of the image is ignored) or off to the side:
        # the multi-view scan covers the whole work area
        print("nothing identified from the look pose: scanning the work area instead")
        t0 = time.time()
        print("scan: " + call(n, "/wrist_map/scan", timeout=240.0).message)
        end = time.time() + 5
        while time.time() < end and got.get("o", {}).get("time", 0) < t0:
            time.sleep(0.1)
        objs = identified()
    if not objs:
        raise SystemExit("no identified object in the map" + (f" named {a.object}" if a.object else ""))
    o = objs[0]
    name, oc = o["object"], cat[o["object"]]
    if o.get("rest") not in (None, "upright", "upside_down"):
        raise SystemExit(f"{name} rests on its {o['rest']}: hover only handles bricks lying flat for now")
    h = float(oc["dims_m"][2])
    T_obj = upright_pose(oc, o["xy"], table_z + h / 2, o["yaw"])
    print(f"target: {name} at ({o['xy'][0]:.3f}, {o['xy'][1]:.3f}) m, yaw {np.degrees(o['yaw']):.0f} deg")
    grasps, why = grasps_from_pose(T_obj, oc, [0.0], R.tcp_yaw(),
                                   gripper_max_width_m=float(cfg["gripper"]["max_width_m"]))
    if not grasps:
        raise SystemExit(f"no grasp: {why}")
    grasps = shift_for_fixed_jaw(grasps, R.gripper_model.held_offset_y,
                                 float(cfg["gripper"].get("fixed_jaw_clearance_m", 0.0)))
    print(f"grasp: {why}; {len(grasps)} candidate(s)")
    width = min(float(cfg["gripper"]["max_width_m"]), grasps[0].width_m + 0.02)
    R.gripper(width, 5.0)
    for i, g in enumerate(grasps):
        T = g.T_base_tcp.copy()
        T[2, 3] += above
        code = R.move_pose(T[:3, 3], matrix_to_quat(T[:3, :3]), scaling=a.scaling)
        if code == -4:      # CONTROL_FAILED: on the real arm usually gravity sag past the goal tolerance -- WHERE is it?
            p, _ = R.tcp_pose()
            err = float(np.linalg.norm(p - T[:3, 3]))
            print(f"  (controller reported a miss; the TCP is {err * 1000:.1f} mm from the target)")
            code = 1 if err < 0.012 else code
        print(f"  candidate {i + 1}: TCP to ({T[0, 3]:.3f}, {T[1, 3]:.3f}, {T[2, 3]:.3f}) m, yaw "
              f"{np.degrees(np.arctan2(T[1, 0], T[0, 0])):.0f} deg: {'ok' if code == 1 else f'MoveIt {code}'}")
        if code == 1:
            p, _ = R.tcp_pose()
            print(f"HOVERING {above * 100:.1f} cm above the grasp point of {name}; TCP now at "
                  f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m, gripper open {width * 1000:.0f} mm. Holding.")
            return dict(name=name, obj_cfg=oc, grasp=g, above=above, width=width, T_cmd=T)
    print("no candidate reachable: move the brick closer to the robot (the work band is ~12-24 cm in front of "
          "the base) and try again")
    return None


if __name__ == "__main__":
    main()
