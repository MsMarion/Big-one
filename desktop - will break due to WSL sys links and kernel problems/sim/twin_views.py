"""Render the sim TWIN of the plate cell the way the real one was photographed, to check the twin by eye.
The arm goes to the look pose (the real snapshot data/jetson_plate_look.png was taken there), the wrist D405 sits at
the REAL unit's calibrated mount (as reality profile `twin` does), the brick lies where the real one was.

    $env:PNP_SCENE="config/scenes/so101_plate.yaml"; .\\run_isaac.ps1 sim\\twin_views.py
-> data/twin/{d435i,d405,overview_front,overview_side}.png   (then tools/twin_report.py composes the comparison)
"""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import json  # noqa: E402
import os  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from common import REPO, load_yaml, log, shutdown  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402
from scene import (TARGET_DIR, add_textured_plate, build_scene, place_target, reset_scene, set_camera_pose,  # noqa: E402
                   set_pinhole, xyzw_to_wxyz)

from pnp_core.calibration import board_size_m  # noqa: E402

from pnp_core.catalog import load_robot  # noqa: E402
from pnp_core.geometry import look_at_optical, make_transform, matrix_to_quat, quat_to_matrix  # noqa: E402
from pnp_core.wrist import wrist_camera  # noqa: E402

OUT = REPO / "data" / "twin"
LOOK_POSE_DEG = [0.0, 0.0, 0.0, 80.0, 0.0]          # tools/move_to.py 0 0 0 80 0 (the real snapshot's pose)
BRICK = ((0.60, -0.30), 0.0)                         # out of view: the real snapshot had only the board


def save(cam: Camera, name: str):
    rgba = cam.get_rgba()
    if rgba is None or rgba.size == 0:
        log(f"[twin] {name}: no image")
        return
    cv2.imwrite(str(OUT / f"{name}.png"), cv2.cvtColor(rgba[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
    log(f"[twin] wrote data/twin/{name}.png {rgba.shape[1]}x{rgba.shape[0]}")


def overview(path, eye, at, res=(960, 600), f=520.0):
    """A plain USD camera posed by its own xform ops (like the wrist camera: Camera.set_world_pose on a NEW camera
    was ignored by its render product here), wrapped for get_rgba()."""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom
    cam = UsdGeom.Camera.Define(get_current_stage(), path)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, eye)))
    R_usd = look_at_optical(eye, at) @ np.diag([1.0, -1.0, -1.0])      # ROS optical -> USD camera (-z forward, +y up)
    qw, qx, qy, qz = xyzw_to_wxyz(matrix_to_quat(R_usd))
    xf.AddOrientOp().Set(Gf.Quatf(float(qw), float(qx), float(qy), float(qz)))
    set_pinhole(cam.GetPrim(), f, f, res[0], res[1])
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.02, 20.0))
    c = Camera(prim_path=path, name=path.rsplit("/", 1)[-1], resolution=res)
    c.initialize()
    return c


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    scene = os.environ.get("PNP_SCENE", "config/scenes/so101_plate.yaml")
    h = build_scene(scene)
    robot = load_robot()
    w = wrist_camera(robot)
    # the wrist D405 as reality profile `twin` puts it: the real unit's calibrated mount
    c = load_yaml("config/calibration/so101_plate/real_d405.yaml")["T_parent_optical"]
    set_camera_pose(h, "wrist_camera", make_transform(quat_to_matrix(c["quat_xyzw"]), c["xyz"]))
    wcam = Camera(prim_path=h.paths["wrist_camera"], name="d405", resolution=(w["width"], w["height"]))
    wcam.initialize()
    h.camera.initialize()
    # the hand board where the REAL D435i saw it (tools/twin_board_check.py real -> data/twin/board_pose.json)
    bp = REPO / "data" / "twin" / "board_pose.json"
    if bp.exists():
        hb = load_yaml("config/calibration_targets_so101.yaml")["hand_board"]
        w_, h_ = board_size_m(hb)
        T_board = np.array(json.loads(bp.read_text())["T_base_board"])
        add_textured_plate("/World/twin_board", T_board, w_, h_, TARGET_DIR / hb.get("image", "hand_board.png"))
        log(f"[twin] hand board at {np.round(T_board[:3, 3], 4).tolist()} (from the real D435i)")
    reset_scene(h, target_xy=None)
    # two outside views: from the front-right corner of the plate, and over the robot's shoulder toward the D435i
    views = [overview("/World/ov_front", [0.75, -0.50, 0.50], [0.25, 0.0, 0.08], f=700.0),
             overview("/World/ov_side", [-0.30, 0.40, 0.45], [0.35, 0.0, 0.10], f=700.0)]
    place_target(h, BRICK[0], BRICK[1])
    # the other bricks out of the way (the real plate had one 2x4)
    from scene import place_extra
    for i, e in enumerate(h.cfg.get("extra_objects", [])):
        place_extra(h, e["name"], [0.60 + 0.05 * i, -0.45], 0.0)
    names = h.robot.dof_names
    q = h.robot.get_joint_positions()
    # the joints where the REAL arm's model has them: the commanded look pose + its fitted offsets (true = encoder + offset)
    off = load_yaml("config/calibration/so101_plate/joint_offsets_real.yaml").get("offsets_deg") or {}
    arm = {j: np.radians(d + off.get(j, 0.0)) for j, d in zip(robot["arm"]["joints"], LOOK_POSE_DEG)}
    for i, n in enumerate(names):
        if n in arm:
            q[i] = arm[n]
    h.robot.set_joint_positions(q)
    h.robot.apply_action(ArticulationAction(joint_positions=q))
    for _ in range(240):                       # 2 s: settle + let the renderer catch up
        h.world.step(render=True)
    # the REAL arm runs a hold loop (pnp_core/servo_hold.py in so101_bridge): its joints reach the goal despite gravity.
    # The sim's drives sag (~1 deg at the elbow here): do the same -- target += remaining error, a few rounds
    goal = q.copy()
    target = q.copy()
    idx = [i for i, n in enumerate(names) if n in arm]
    for _ in range(4):
        err = goal - h.robot.get_joint_positions()
        target[idx] += err[idx]
        h.robot.apply_action(ArticulationAction(joint_positions=target))
        for _ in range(120):
            h.world.step(render=True)
    # what the sim actually did: the joints after settling, and the D405's world pose (tools/twin_board_check.py
    # compares them with the robot's own kinematics)
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdGeom
    qa = h.robot.get_joint_positions()
    M = UsdGeom.Xformable(get_current_stage().GetPrimAtPath(h.paths["wrist_camera"])).ComputeLocalToWorldTransform(0)
    T_usd = np.array(M).T                                   # USD row-vector convention -> column 4x4
    T_opt = T_usd.copy()
    T_opt[:3, :3] = T_usd[:3, :3] @ np.diag([1.0, -1.0, -1.0])   # USD camera (-z forward) -> ROS optical
    (OUT / "sim_state.json").write_text(json.dumps({
        "joints_deg": {n: round(float(np.degrees(v)), 3) for n, v in zip(names, qa)},
        "joints_cmd_deg": {n: round(float(np.degrees(arm[n])), 3) for n in arm},
        "T_base_d405": T_opt.round(6).tolist()}, indent=1))
    log(f"[twin] joints after settling (deg): {[round(float(np.degrees(v)), 2) for v in qa]}")
    save(h.camera, "d435i")
    save(wcam, "d405")
    save(views[0], "overview_front")
    save(views[1], "overview_side")
    log("[twin] done")
    return 0


if __name__ == "__main__":
    shutdown(app, main())
