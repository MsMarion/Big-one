"""Find a brick, hover over it, then GRAB it: a close look from the hover, a straight descent, close, check, lift.
Run on the robot side (the Jetson):  docker exec jetson-robot-1 bash -ic "python3 /repo/tools/grab.py"
    (the options of tools/hover_over.py, plus)
    --no-close        stop at the grasp height with the gripper open (look, then decide)
    --force 0.3       gripper effort (the robot profile's units; ~N m at the jaw)

Why a CLOSE LOOK: how far down to go is the one thing that must not be wrong (too far = the jaws drive into the desk).
The map's heights rest on the whole arm chain (joint angles, offsets, sag) and on the configured desk height -- the
camera put the real desk ~12 mm below the configured one. But the D405 is bolted to the gripper: its points expressed
in the TCP frame need ONLY the wrist calibration, no joint angle. From the hover it sees the brick's top and the desk,
so "the top is N mm below the TCP, its center M mm to the side" is measured directly. The descent goes there.

Centering: the close look steers -- shift at the hover height by what it measures, look again, until the brick is
within 3 mm of where the grasp wants it (max 3 shifts, each <= 40 mm). The blob must be as tall as the catalog brick
(+-4 mm), else it stops. Then down to 1 cm above, the last centimetre at 5 %, close, read the jaw width (holding ~ the
brick's width; ~0 = missed), lift. The torque stays on at the end.
"""
import argparse
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, "/repo/tools")
from hover_over import add_args, hover  # noqa: E402
from pnp_core.catalog import load_robot, load_scene, load_yaml  # noqa: E402
from pnp_core.height_map import HeightMap, classify, fit_desk_plane, level_to_plane  # noqa: E402
from pnp_core.reorient import rest_poses  # noqa: E402
from pnp_core.geometry import make_transform, matrix_to_quat, quat_to_matrix  # noqa: E402
from pnp_core.perception_math import depth_to_meters  # noqa: E402
from pnp_manipulation.robot_interface import RobotInterface  # noqa: E402


def close_look(n, R, name, expect_xy_tcp, frames=5, half=0.07):
    """The brick, seen from where the arm is now, through the FLAT-MAP machinery (desk plane fit, height band, blobs,
    catalog match) -- then expressed in the TCP frame. -> (x, y, z_top, desk_z, fully_seen) in the TCP frame (z down) or None.

    Why the base-frame detour cancels the arm's errors: the points go camera -> base and the TCP -> base through the SAME
    joint chain at the same moment; only the wrist calibration (camera -> gripper) remains. The height band (nothing
    more than brick + 6 mm above the desk) drops the open jaws hanging 3-5 cm above it."""
    cat = load_yaml("config/objects.yaml")
    h_cat = float(cat[name]["dims_m"][2])
    scene = load_scene()
    t = scene["table"]
    table_z = t["center"][2] + t["size"][2] / 2
    got, info = [], {}
    sub_d = n.create_subscription(Image, "/wrist_camera/camera/aligned_depth_to_color/image_raw",
                                  lambda m: got.append(m), qos_profile_sensor_data)
    sub_i = n.create_subscription(CameraInfo, "/wrist_camera/camera/color/camera_info",
                                  lambda m: info.__setitem__("k", m), qos_profile_sensor_data)
    end = time.time() + 5
    while (len(got) < frames or "k" not in info) and time.time() < end:
        time.sleep(0.05)
    n.destroy_subscription(sub_d)
    n.destroy_subscription(sub_i)
    if not got or "k" not in info:
        print("close look: no wrist depth")
        return None
    k = info["k"].k
    pos, Rb = R.tcp_pose()
    T_b_tcp = make_transform(Rb, pos)
    c_base = (T_b_tcp @ np.array([expect_xy_tcp[0], expect_xy_tcp[1], 0.0, 1.0]))[:2]
    fit = None
    hm = HeightMap((c_base[0] - half, c_base[0] + half), (c_base[1] - half, c_base[1] + half), 0.002, table_z)
    for m in got[-frames:]:
        z = depth_to_meters(np.frombuffer(m.data, np.uint16).reshape(m.height, m.width), m.encoding)
        vs, us = np.mgrid[0:z.shape[0], 0:z.shape[1]]
        ok = (z > 0.06) & (z < 0.35)
        p = np.column_stack([(us[ok] - k[2]) * z[ok] / k[0], (vs[ok] - k[5]) * z[ok] / k[4], z[ok]])
        tt = R.tf.lookup_transform(R.base, m.header.frame_id, rclpy.time.Time()).transform
        T = make_transform(quat_to_matrix([tt.rotation.x, tt.rotation.y, tt.rotation.z, tt.rotation.w]),
                           [tt.translation.x, tt.translation.y, tt.translation.z])
        P = p @ T[:3, :3].T + T[:3, 3]
        f = fit_desk_plane(P, table_z)
        if f is None:
            continue
        fit = f
        P = level_to_plane(P, fit[0], fit[1], table_z)
        P = P[P[:, 2] < table_z + h_cat + 0.006]              # the jaws hang far higher: gone
        hm.add(P)
    full = [o for o in hm.objects() if classify(o, cat, [name], rest_poses).get("object") == name]
    # partly hidden (under the gripper / at the image edge): the brick's height and width, but shorter -- enough to steer
    dims = sorted(cat[name]["dims_m"][:2], reverse=True)
    part = [o for o in hm.objects() if abs(o.height - h_cat) < 0.004 and abs(o.footprint[1] - dims[1]) < 0.006
            and 0.5 * dims[0] <= o.footprint[0] <= dims[0] + 0.006]
    if not part:
        # mostly hidden under the gripper (the hover can land ~2 cm off): any blob as TALL as the brick, not bigger,
        # is enough to steer toward -- the next look sees more of it; the descent still needs the whole brick
        part = [o for o in hm.objects() if abs(o.height - h_cat) < 0.004 and o.footprint[0] <= dims[0] + 0.006
                and o.footprint[0] >= 0.006]
    objs = full or part
    if not objs or fit is None:
        found = [f"{o.footprint[0] * 1000:.0f}x{o.footprint[1] * 1000:.0f}x{o.height * 1000:.0f} mm at "
                 f"({o.center_xy[0]:.3f}, {o.center_xy[1]:.3f})" for o in hm.objects()]
        print(f"close look: no {name} in view (window center ({c_base[0]:.3f}, {c_base[1]:.3f}), coverage "
              f"{hm.coverage * 100:.0f}%, blobs: {found or 'none'})")
        return None
    o = min(objs, key=lambda o: np.hypot(o.center_xy[0] - c_base[0], o.center_xy[1] - c_base[1]))
    # the desk under the brick in the RAW (unlevelled) base frame: the fitted plane at the brick's x, y
    zd = -(fit[1] + fit[0][0] * o.center_xy[0] + fit[0][1] * o.center_xy[1]) / fit[0][2]
    desk_base = np.array([o.center_xy[0], o.center_xy[1], zd])
    top_base = desk_base + np.array([0.0, 0.0, o.height])
    inv = np.linalg.inv(T_b_tcp)
    top_tcp, desk_tcp = (inv @ np.append(top_base, 1.0))[:3], (inv @ np.append(desk_base, 1.0))[:3]
    return float(top_tcp[0]), float(top_tcp[1]), float(top_tcp[2]), float(desk_tcp[2]), bool(full)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    ap.add_argument("--no-close", action="store_true")
    ap.add_argument("--force", type=float, default=0.3)
    ap.add_argument("--extra-depth", type=float, default=0.003,
                    help="go this much deeper than the measured grasp height (m). 2026-10-11, by eye: the jaw tips "
                         "stopped 12.5 mm above the desk over an 11.3 mm brick -- just above its top, closing on air")
    ap.add_argument("--sag-comp", action="store_true",
                    help="add (commanded - measured) to the final command: only WITHOUT the bridge's hold loop (hold_ki 0); "
                         "with it the joints reach their goal and this double-corrects (2026-10-11 plate: +14 mm -> miss)")
    ap.add_argument("--look-offset", type=float, default=0.03, help="TCP +x shift before the close look (m)")
    a = ap.parse_args()
    rclpy.init()
    n = rclpy.create_node("grab")
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    R = RobotInterface(n, load_robot())
    h = hover(n, R, a)
    if h is None:
        return 1
    g, oc = h["grasp"], h["obj_cfg"]
    depth = float(oc["grasp"]["depth_below_top_m"])
    p_o = g.T_tcp_obj[:3, 3]                       # the object's center in the TCP frame AT the grasp
    # expected at the hover (TCP `above` higher): the brick's top at above + depth below the TCP, centered at p_o
    # LOOK FROM THE SIDE: the D405 sits ~7 cm behind the TCP (TCP -x) looking forward-down; right under the jaws the
    # gripper hides the desk. So: hover with the brick ~3 cm toward the camera, where it is in full view, measure it
    # there (TCP frame: wrist calibration only, no joint angles), then ONE move puts the TCP over the grasp point.
    h_cat = float(oc["dims_m"][2])
    side = np.array([a.look_offset, 0.0])
    # every move is computed from the COMMANDED pose (+ corrections), never from the measured one: the arm sags a few mm
    # below each command, and chaining off the measured pose walked the gripper ~8 mm DOWN per step
    cmd_p, cmd_R = h["T_cmd"][:3, 3].copy(), h["T_cmd"][:3, :3].copy()
    cmd_q = matrix_to_quat(cmd_R)
    cmd_p = cmd_p + cmd_R @ np.array([side[0], side[1], 0.0])
    code = R.move_pose(cmd_p, cmd_q, linear=True, scaling=0.1)
    if code not in (1, -4):
        print(f"STOP: the side-look move failed (MoveIt {code})")
        return 1
    # center the brick on the side-look spot (p_o - side) in steps: the first look searches wide (the map can be ~3 cm
    # off from the look pose), the next ones confirm with the brick fully in view
    for it in range(4):
        time.sleep(0.8)
        cl = close_look(n, R, h['name'], p_o[:2] - side, half=0.07 if it == 0 else 0.045)
        if cl is None:
            print("STOP: no close look, not descending")
            return 1
        cx, cy, z_top, desk, whole = cl
        tall = desk - z_top
        err = np.array([cx, cy]) - (p_o[:2] - side)              # brick vs the side-look spot (TCP frame)
        print(f"close look {it + 1} ({'whole brick' if whole else 'PART of it'}): top {z_top * 1000:.1f} mm below the "
              f"TCP, {tall * 1000:.1f} mm tall, {np.linalg.norm(err) * 1000:.1f} mm from the side-look spot")
        if np.linalg.norm(err) < 0.004 and whole:
            break
        if np.linalg.norm(err) > 0.07 or it == 3:
            print("STOP: can't center the brick in the side view")
            return 1
        cmd_p = cmd_p + cmd_R @ np.array([err[0], err[1], 0.0])
        code = R.move_pose(cmd_p, cmd_q, linear=True, scaling=0.1)
        if code not in (1, -4):
            print(f"STOP: centering move failed (MoveIt {code})")
            return 1
    if abs(tall - h_cat) > 0.004:
        print(f"STOP: that blob is {tall * 1000:.1f} mm tall, the {h['name']} is {h_cat * 1000:.1f}: not descending")
        return 1
    print(f"brick centered in the side view: top {z_top * 1000:.1f} mm below the TCP, {tall * 1000:.1f} mm tall")
    # the grasp TCP in the CURRENT TCP frame: the measured top + `depth`, laterally so the brick sits where p_o says
    target_tcp = np.array([cx - p_o[0], cy - p_o[1], z_top + depth + a.extra_depth])
    pos, Rb = R.tcp_pose()
    # where the grasp IS (measured frame) -> what to COMMAND. The bridge's hold loop (pnp_core/servo_hold.py) brings
    # the joints to their goal, so the measured pose IS where commands land: command it directly. Without the hold
    # loop the arm sagged below each command: --sag-comp adds that offset (command - actual) back.
    sag = cmd_p - pos
    goal = pos + Rb @ target_tcp + (sag if a.sag_comp else 0.0)
    quat = cmd_q
    above1 = goal + cmd_R @ np.array([0.0, 0.0, -0.01])                 # 1 cm above the grasp (TCP z points down)
    print(f"commanded - measured here: ({sag[0] * 1000:+.1f}, {sag[1] * 1000:+.1f}, {sag[2] * 1000:+.1f}) mm "
          f"({'added to the goal' if a.sag_comp else 'not used: the hold loop closes it'})")
    # MoveIt's desk where the close look SAW it (in the arm's own frame, the real desk sits ~12 mm below the configured
    # one: the last centimetre was "gripper - table" blocked although the jaws would stop 9 mm above it)
    desk_z = float((pos + Rb @ np.array([cx, cy, desk]))[2])
    t = load_scene()["table"]
    R.add_box("table", [t["center"][0], t["center"][1], desk_z - 0.004 - t["size"][2] / 2], t["size"])
    print(f"desk measured at z {desk_z * 1000:+.1f} mm (configured {(t['center'][2] + t['size'][2] / 2) * 1000:+.1f}): "
          f"MoveIt's desk moved there")
    print(f"descent: {target_tcp[2] * 1000:.1f} mm down, lateral correction ({target_tcp[0] * 1000:+.1f}, "
          f"{target_tcp[1] * 1000:+.1f}) mm")
    code = R.move_pose(above1, quat, linear=True, scaling=0.1)
    print(f"  to 1 cm above: {'ok' if code == 1 else f'MoveIt {code}'}")
    if code != 1:
        return 1
    code = R.move_pose(goal, quat, linear=True, scaling=0.05)
    print(f"  to the grasp: {'ok' if code == 1 else f'MoveIt {code}'}")
    if code != 1:
        return 1
    if a.no_close:
        print("at the grasp height, gripper open (--no-close): holding")
        return 0
    w = R.close_on_object(a.force)
    exp_w = float(g.width_m)
    held = abs(w - exp_w) < 0.006
    print(f"closed: jaw width {w * 1000:.1f} mm (brick {exp_w * 1000:.1f} mm): {'HOLDING' if held else 'MISSED?'}")
    up = goal + cmd_R @ np.array([0.0, 0.0, -float(R.cfg.get('grasp_motion', {}).get('lift_height_m', 0.045))])
    code = R.move_pose(up, quat, linear=True, scaling=0.1)
    w2 = R.gripper_width()
    print(f"lift: {'ok' if code == 1 else f'MoveIt {code}'}; jaw width now {w2 * 1000:.1f} mm "
          f"({'still holding' if abs(w2 - exp_w) < 0.006 else 'EMPTY'})")
    ex.shutdown()
    rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
