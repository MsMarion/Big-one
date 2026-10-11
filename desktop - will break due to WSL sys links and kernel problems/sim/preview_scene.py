"""Build the scene headless, settle physics, save what the camera sees, and run sanity checks.

Checks (each prints PASS/FAIL; exit code 1 if any fail):
  1. The can settles upright on the table (no tipping / falling through).
  2. Frame alignment: projecting the can's TRUE position through our pinhole model (config intrinsics
     + config mount) lands on the can in the rendered image. This catches USD-vs-ROS camera axis
     mistakes and intrinsics mismatches before any ROS or ML is involved.
  3. Depth at that pixel matches the true camera-to-can distance.

Outputs: data/preview/rgb.png, data/preview/depth.png (colorized), data/preview/overlay.png

Run:  .\\run_isaac.ps1 sim\\preview_scene.py
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import os

import cv2
import numpy as np
from common import REPO, load_yaml, log, shutdown
from pnp_core.catalog import load_fixed_camera  # noqa: E402
from scene import build_scene, reset_scene

from pnp_core.geometry import transform_points
from pnp_core.perception_math import Intrinsics

OUT = REPO / "data" / "preview"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    h = build_scene(os.environ.get("PNP_SCENE", "config/scenes/basic.yaml"))   # the twin: PNP_SCENE=config/scenes/so101_plate.yaml
    # A third-person "overview" camera, to see the workcell incl. the D435i rig (data/preview/overview.png)
    from isaacsim.sensors.camera import Camera
    from pnp_core.geometry import look_at_optical, matrix_to_quat, camera_mount_T
    from scene import set_pinhole, xyzw_to_wxyz
    overview = Camera(prim_path="/World/overview_cam", name="overview", resolution=(960, 600))
    set_pinhole(overview.prim, 420.0, 420.0, 960, 600)          # wide angle (~98 deg horizontal)
    # where it looks from / at: env PNP_OVERVIEW_EYE / PNP_OVERVIEW_AT ("x,y,z" in the robot base frame)
    eye = [float(v) for v in os.environ.get("PNP_OVERVIEW_EYE", "1.7,-0.9,1.15").split(",")]
    at = [float(v) for v in os.environ.get("PNP_OVERVIEW_AT", "0.55,0.15,0.15").split(",")]
    overview.set_world_pose(position=np.array(eye), camera_axes="ros",
                            orientation=xyzw_to_wxyz(matrix_to_quat(look_at_optical(eye, at))))
    overview.initialize()
    overview.set_clipping_range(0.02, 20.0)    # Isaac's default near clip (~1 m) cut away everything close
    h.camera.initialize()
    h.camera.add_distance_to_image_plane_to_frame()
    reset_scene(h, target_xy=h.cfg["target"]["default_xy"])
    for _ in range(120):                       # 1 s of physics at 120 Hz, rendering every few steps
        h.world.step(render=True)

    cam = load_fixed_camera()
    obj = load_yaml("config/objects.yaml")[h.cfg["target"]["object"]]
    K_isaac = h.camera.get_intrinsics_matrix()
    log(f"[check] Isaac camera intrinsics fx={K_isaac[0, 0]:.1f} fy={K_isaac[1, 1]:.1f} "
        f"cx={K_isaac[0, 2]:.1f} cy={K_isaac[1, 2]:.1f} (config fx={cam['fx']})")
    prim = h.camera.prim
    log(f"[debug] usd focalLength={prim.GetAttribute('focalLength').Get()} "
        f"hAperture={prim.GetAttribute('horizontalAperture').Get()} vAperture={prim.GetAttribute('verticalAperture').Get()} "
        f"rgba shape={h.camera.get_rgba().shape}")
    p_ros, q_ros = h.camera.get_world_pose(camera_axes="ros")
    log(f"[debug] camera pose (ros axes) p={np.round(p_ros, 3)} q_wxyz={np.round(q_ros, 3)}")
    ok_k = abs(K_isaac[0, 0] - cam["fx"]) < 1.0 and abs(K_isaac[1, 1] - cam["fy"]) < 1.0
    log(f"[check] intrinsics match config -> {'PASS' if ok_k else 'FAIL'}")
    pos, _ = h.target.get_world_pose()
    ok = ok_k

    # 1. upright & resting on the table
    expected_z = obj["dims_m"][2] / 2
    upright = abs(pos[2] - expected_z) < 0.01
    log(f"[check] can position {np.round(pos, 4)}; expected z ~ {expected_z:.4f} -> {'PASS' if upright else 'FAIL'}")
    ok &= upright

    # 2. projection of the true center vs. where the can is in the image
    rgba = h.camera.get_rgba()
    depth = h.camera.get_current_frame()["distance_to_image_plane"]
    if rgba is None or rgba.size == 0 or depth is None:
        log("[check] camera returned no data -> FAIL")
        return 1
    rgb = rgba[:, :, :3].astype(np.uint8)
    K = Intrinsics(cam["fx"], cam["fy"], cam["cx"], cam["cy"])
    T_base_cam = camera_mount_T(cam["mount"])
    p_cam = transform_points(np.linalg.inv(T_base_cam), pos[None, :])
    u, v = K.project(p_cam)[0]
    log(f"[check] true can center projects to pixel ({u:.1f}, {v:.1f}), depth {p_cam[0, 2]:.3f} m")

    in_image = 0 <= u < cam["width"] and 0 <= v < cam["height"]
    ok &= in_image
    if in_image:
        # The can is much closer than the table behind it: check the depth step at the projected pixel.
        d = float(depth[int(v), int(u)])
        dist_ok = abs(d - p_cam[0, 2]) < obj["dims_m"][0]       # within one can-diameter of the axis
        log(f"[check] rendered depth at that pixel {d:.3f} m vs true {p_cam[0, 2]:.3f} m -> {'PASS' if dist_ok else 'FAIL'}")
        ok &= dist_ok
    else:
        log("[check] true center projects OUTSIDE the image -> FAIL")

    # outputs
    cv2.imwrite(str(OUT / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    dvis = np.nan_to_num(depth, nan=0.0, posinf=0.0)
    dvis = cv2.applyColorMap(cv2.convertScaleAbs(dvis, alpha=255.0 / 2.0), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(OUT / "depth.png"), dvis)
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if in_image:
        cv2.drawMarker(overlay, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.imwrite(str(OUT / "overlay.png"), overlay)
    ov = overview.get_rgba()
    if ov is not None and ov.size:
        cv2.imwrite(str(OUT / "overview.png"), cv2.cvtColor(ov[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR))
    log(f"[preview] images written to {OUT}")
    log(f"[preview] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    shutdown(app, main())
