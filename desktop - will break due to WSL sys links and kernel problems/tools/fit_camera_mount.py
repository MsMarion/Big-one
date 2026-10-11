"""Where does the D435i's mount stand on the plate? Fit the CAD mount (robots/SO101_UVC_D435i_URDF, the user's Fusion
export) to the camera's CALIBRATED pose: the mast's x, y and yaw on the board (its base sits flat on the board) and
the tilt joint, such that the D435i's color optical frame lands on the calibration.

    python3 /repo/tools/fit_camera_mount.py [scene] [--write]     (default scene: so101_plate_tags)

The URDF has no optical frame: it is placed in the camera body from Intel's realsense2_description (_d435.urdf.xacro):
the color imager 15 mm beside the depth origin (the left IR imager), which sits 17.5 mm from the body's centre, ~4.3 mm
behind the front glass. Which face is the front (the lens side) comes from the README: at tilt 0 the camera looks
21.7 deg below horizontal. --write puts the result into the scene file (camera_rig), so the sim shows the stand where
the camera really is; the printed placement (cm / inches from the robot base) is for checking with a tape measure.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/repo")
from pnp_core.calibration import pose_error  # noqa: E402
from pnp_core.catalog import load_fixed_camera, load_yaml  # noqa: E402
from pnp_core.geometry import camera_mount_T, make_transform  # noqa: E402
from pnp_core.urdf_visuals import axis_R, load_urdf_visuals  # noqa: E402
from pnp_core.urdf_visuals import link_T as _link_T  # noqa: E402

URDF = Path("/repo/robots/SO101_UVC_D435i_URDF/so101_uvc_d435i_mount.urdf")
COLOR_FROM_CENTER = 0.0175 + 0.015          # m along the width, toward the camera's LEFT (Intel: depth py + color)
LENS_BEHIND_FRONT = 0.0043                  # m (d435_zero_depth_to_glass + glass_to_front)


def load_mount(path=URDF):
    return load_urdf_visuals(path)


def link_T(link, joints, tilt):
    """mount base -> link, the tilt joint at `tilt` rad (added to the URDF's zero = the export pose)."""
    return _link_T(link, joints, {"d435i_tilt": tilt})


def optical_in_camera_link():
    """The color optical frame in d435i_camera_link (the STL's frame: x = width 90 mm, y = height 25 mm, z = depth,
    body z in [-25.1, 0] mm). The front (lens side) is the face that, at tilt 0, looks 21.7 deg below horizontal:
    decided in fit() from the geometry; returned for both candidates."""
    out = {}
    for front in (+1, -1):                       # lens at z = 0 (front +z) or at z = -25.1 mm (front -z)
        f = np.array([0, 0, front], float)
        face_z = 0.0 if front > 0 else -0.0251
        for up in (+1, -1):
            u = np.array([0, up, 0], float)
            left = np.cross(u, f)                # x forward, y left, z up  ->  left = up x forward
            p = np.array([0, 0, face_z]) - f * LENS_BEHIND_FRONT + left * COLOR_FROM_CENTER
            R = np.column_stack([-left, -u, f])  # optical: x right, y down, z forward
            out[(front, up)] = make_transform(R, p)
    return out


def fit(scene):
    base, joints, _ = load_mount()
    cam = load_fixed_camera(scene)
    T_target = camera_mount_T(cam["mount"])                       # base_link <- color optical (the calibration)
    t = scene["table"]
    z_board = t["center"][2] + t["size"][2] / 2
    T_cl0 = link_T("d435i_camera_link", joints, 0.0)
    # the lens side: the candidate whose optical axis at tilt 0 is ~21.7 deg below horizontal and whose image "down"
    # points down (camera upright)
    cands = []
    for key, T_co in optical_in_camera_link().items():
        R = (T_cl0 @ T_co)[:3, :3]
        down = math.degrees(math.asin(-R[2, 2]))
        cands.append((abs(down - 21.709) + (0 if R[2, 1] < 0 else 90), key, down))
    _, key, down0 = min(cands)
    T_co = optical_in_camera_link()[key]
    print(f"lens side: camera-link {'+' if key[0] > 0 else '-'}z, up {'+' if key[1] > 0 else '-'}y; at the export tilt "
          f"the camera looks {down0:.1f} deg below horizontal")

    def T_opt(p):
        x, y, yaw, tilt = p
        T_base_mount = make_transform(axis_R([0, 0, 1], yaw), [x, y, z_board])
        return T_base_mount @ link_T("d435i_camera_link", joints, tilt) @ T_co

    def resid(p):
        T = T_opt(p)
        # the whole rotation matrix (a small-angle vector reads a 180 deg flip as 0); 1 rad ~ 500 mm: 1 deg weighs
        # like the ~9 mm it moves a point half a metre away
        return np.concatenate([(T[:3, 3] - T_target[:3, 3]) * 1000.0, (T[:3, :3] - T_target[:3, :3]).ravel() * 500.0])
    from scipy.optimize import least_squares
    best = None
    for yaw0 in np.radians(np.arange(0, 360, 45)):
        for tilt0 in np.radians([-40, -20, 0, 20, 40]):
            T0 = make_transform(axis_R([0, 0, 1], yaw0), [0, 0, 0]) @ link_T("d435i_camera_link", joints, tilt0) @ T_co
            xy0 = T_target[:2, 3] - T0[:2, 3]
            r = least_squares(resid, [xy0[0], xy0[1], yaw0, tilt0])
            if best is None or r.cost < best.cost:
                best = r
    x, y, yaw, tilt = best.x
    mm, deg = pose_error(T_opt(best.x), T_target)
    T_pivot = make_transform(axis_R([0, 0, 1], yaw), [x, y, z_board]) @ joints["d435i_tilt_link"][1]
    return {"x": x, "y": y, "yaw": yaw, "tilt": tilt, "err_mm": mm, "err_deg": deg, "pivot": T_pivot[:3, 3],
            "down_deg": math.degrees(math.asin(-T_opt(best.x)[2, 2])), "z_board": z_board}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    name = args[0] if args else "so101_plate_tags"
    path = Path(f"/repo/config/scenes/{name}.yaml")
    scene = load_yaml(str(path))
    r = fit(scene)
    yaw_deg = math.degrees(r["yaw"]) % 360
    print(f"mount base (its URDF origin) on the board at x {r['x'] * 100:.1f} cm / {r['x'] / 0.0254:.2f} in, "
          f"y {r['y'] * 100:+.1f} cm / {r['y'] / 0.0254:+.2f} in from the robot base, turned {yaw_deg:.1f} deg")
    print(f"tilt joint {math.degrees(r['tilt']):+.1f} deg from the export pose -> the camera looks "
          f"{r['down_deg']:.1f} deg below horizontal; pivot at {np.round(r['pivot'] * 100, 1).tolist()} cm")
    print(f"fit vs the calibration: {r['err_mm']:.1f} mm / {r['err_deg']:.2f} deg (what the CAD + Intel's lens offsets "
          f"can't explain: the camera's roll in the fork, print / assembly tolerance)")
    if "--write" in sys.argv:
        text = path.read_text()
        block = (f"camera_rig:                       # the D435i's stand: the user's Fusion export, fitted to the calibrated "
                 f"camera\n  # (tools/fit_camera_mount.py: residual {r['err_mm']:.1f} mm / {r['err_deg']:.2f} deg)\n"
                 f"  urdf: robots/SO101_UVC_D435i_URDF/so101_uvc_d435i_mount.urdf\n"
                 f"  xy: [{r['x']:.4f}, {r['y']:.4f}]\n  yaw_deg: {yaw_deg:.2f}\n  tilt_rad: {r['tilt']:.4f}\n\n")
        import re
        text = re.sub(r"camera_rig:.*?\n\n", "", text, flags=re.S)
        text = text.replace("\ncamera:\n", "\n" + block + "camera:\n", 1)
        path.write_text(text)
        print(f"written: camera_rig in {path.relative_to('/repo')}")


if __name__ == "__main__":
    main()
