"""Does the sim TWIN see what the real cell sees? One printed board, two cameras, real vs sim, in pixels.

  1. python3 tools/twin_board_check.py real       the REAL snapshot (data/jetson_plate_look.png: left the D435i,
     right the wrist D405, the arm at the look pose 0 0 0 80 0): the hand board's pose on the plate from the
     calibrated D435i -> data/twin/board_pose.json, plus every corner the real cameras saw
  2. .\\run_isaac.ps1 sim\\twin_views.py           renders the twin with that board and the arm at the same pose
  3. python3 tools/twin_board_check.py compare    the same corners in the sim images: pixel differences per camera
How to read it:
  D435i  ~0 px by construction (the board was placed from this camera): a check of the camera MIRROR itself
         (intrinsics, the principal point, the pose convention). Anything above ~1 px is a twin bug.
  D405   the arm chain (joint offsets, link lengths, sag, the D405 mount) of the twin vs the real arm: px -> mm at
         the board's distance. This is how far the twin's model of the real arm is off.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pnp_core.calibration import detect_board  # noqa: E402
from pnp_core.catalog import load_yaml  # noqa: E402
from pnp_core.geometry import make_transform, quat_to_matrix  # noqa: E402

TWIN = REPO / "data" / "twin"
SCENE = load_yaml("config/scenes/so101_plate.yaml")
BOARD = load_yaml("config/calibration_targets_so101.yaml")["hand_board"]


def K_of(c):
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])


K435 = K_of(SCENE["camera"])
K405 = K_of(SCENE["wrist_camera_intrinsics"])
D405_REAL = np.array([-0.05385, 0.058, 0.00069, 0.0003, -0.01948])      # the real D405's camera_info (sim: none)


def corners(img, K, D=None):
    """-> ({board point (mm, rounded): pixel}, T_cam_board) of a board detection, undistorted to the pinhole K."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    T, n, err, pts = detect_board(gray, BOARD, K, np.zeros(5) if D is None else D, return_points=True)
    if T is None:
        return {}, None
    obj, px = np.asarray(pts[0], float), np.asarray(pts[1], float).reshape(-1, 2)
    if D is not None and np.any(D):
        px = cv2.undistortPoints(px.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    return {tuple(np.round(o * 1000).astype(int)): p for o, p in zip(obj, px)}, T


def split(path):
    im = cv2.imread(str(path))
    return im[:, :640], im[:360, 640:]           # tools/snap.py's side-by-side: D435i 640x480 | D405 640x360


def real():
    d435, d405 = split(REPO / "data" / "jetson_plate_look.png")
    c435, T_cam_board = corners(d435, K435)
    if T_cam_board is None:
        print("no board in the real D435i image")
        return 1
    cal = load_yaml("config/calibration/so101_plate/real_d435i.yaml")["T_parent_optical"]
    T_base_board = make_transform(quat_to_matrix(cal["quat_xyzw"]), cal["xyz"]) @ T_cam_board
    c405, _ = corners(d405, K405, D405_REAL)
    TWIN.mkdir(parents=True, exist_ok=True)
    out = {"T_base_board": T_base_board.round(6).tolist(),
           "real": {"d435i": {str(k): v.tolist() for k, v in c435.items()},
                    "d405": {str(k): v.tolist() for k, v in c405.items()}}}
    (TWIN / "board_pose.json").write_text(json.dumps(out, indent=1))
    c = T_base_board[:3, 3]
    print(f"board (corner origin) at {np.round(c, 4).tolist()} m in the base frame, from the real D435i "
          f"({len(c435)} corners); the real D405 saw {len(c405)} corners -> data/twin/board_pose.json")
    return 0


def compare():
    ref = json.loads((TWIN / "board_pose.json").read_text())["real"]
    rows = []
    for cam, img, K in (("d435i", TWIN / "d435i.png", K435), ("d405", TWIN / "d405.png", K405)):
        sim, T = corners(cv2.imread(str(img)), K)
        real_c = {k: np.array(v) for k, v in ref[cam].items()}
        both = [k for k in map(str, sim) if k in real_c]
        if not both:
            rows.append(f"  {cam}: no common corners (sim saw {len(sim)}, real {len(real_c)})")
            continue
        simd = {str(k): v for k, v in sim.items()}
        d = np.array([simd[k] - real_c[k] for k in both])
        err = np.linalg.norm(d, axis=1)
        dist = float(np.linalg.norm(T[:3, 3]))
        mm = err.mean() / K[0, 0] * dist * 1000
        rows.append(f"  {cam}: {len(both)} corners, sim - real = {err.mean():.1f} px mean, {err.max():.1f} max, "
                    f"shift ({d[:, 0].mean():+.1f}, {d[:, 1].mean():+.1f}) px  ~ {mm:.1f} mm at {dist:.2f} m")
    print("twin vs real, the same board corners:\n" + "\n".join(rows))
    return 0


def chain():
    """The sim's D405 pose (sim/twin_views.py -> data/twin/sim_state.json) vs the ROBOT's own model of it: the URDF
    with the real joint offsets folded in, at the commanded look pose, + the real D405 mount. They must agree."""
    from pnp_core import robot_description as rdesc
    from pnp_core.catalog import load_robot, repo_root
    from pnp_core.urdf_kin import Chain
    st = json.loads((TWIN / "sim_state.json").read_text())
    robot = load_robot("so101")
    off = load_yaml("config/calibration/so101_plate/joint_offsets_real.yaml")["offsets_deg"]
    ch = Chain(rdesc.urdf(robot, repo_root(), "mock", offsets={j: np.radians(v) for j, v in off.items()}),
               robot["frames"]["base"], "gripper_link")
    cal = load_yaml("config/calibration/so101_plate/real_d405.yaml")["T_parent_optical"]
    T_mount = make_transform(quat_to_matrix(cal["quat_xyzw"]), cal["xyz"])
    look = [0.0, 0.0, 0.0, 80.0, 0.0]
    T_model = ch.fk(np.radians(look)) @ T_mount
    T_sim = np.array(st["T_base_d405"])
    q_sim = [st["joints_deg"][j] - off.get(j, 0.0) for j in robot["arm"]["joints"]]   # sim joints as "encoder" angles
    T_simq = ch.fk(np.radians(q_sim)) @ T_mount
    def diff(A, B):
        return (np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000,
                np.degrees(np.arccos(np.clip((np.trace(A[:3, :3].T @ B[:3, :3]) - 1) / 2, -1, 1))))
    print("sim joints after settling - commanded (deg):",
          [round(st["joints_deg"][j] - st["joints_cmd_deg"][j], 2) for j in robot["arm"]["joints"]])
    print("D405: sim prim vs the robot's model at the commanded pose: %.1f mm / %.2f deg" % diff(T_sim, T_model))
    print("D405: sim prim vs the robot's model at the sim's ACTUAL joints: %.1f mm / %.2f deg (chain / mount mismatch)"
          % diff(T_sim, T_simq))
    print("   sim at", np.round(T_sim[:3, 3], 4).tolist(), " model at", np.round(T_model[:3, 3], 4).tolist())
    return 0


if __name__ == "__main__":
    sys.exit({"real": real, "compare": compare, "chain": chain}[sys.argv[1] if len(sys.argv) > 1 else "real"]())
