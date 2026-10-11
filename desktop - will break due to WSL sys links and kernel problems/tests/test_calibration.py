import cv2
import numpy as np

from pnp_core.calibration import (apriltag_image, board_center_T, board_image, board_size_m, detect_board,
                                  detect_tags, mount_T, pose_error, residuals, solve_hand_eye)
from pnp_core.catalog import load_yaml
from pnp_core.geometry import look_at_optical, make_transform

TGT = load_yaml("config/calibration_targets.yaml")
K = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1]])


def rot(axis, deg):
    a = np.radians(deg)
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def render(img_board, px_per_m, T_cam_plane, K, size=(480, 640)):
    """Project a planar texture (its pixel (u, v) = plane point (u/px_per_m, v/px_per_m, 0)) into a camera image."""
    h, w = img_board.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    pts = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], float) / px_per_m
    pc = pts @ T_cam_plane[:3, :3].T + T_cam_plane[:3, 3]
    dst = (pc @ K.T)[:, :2] / pc[:, 2:3]
    H = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    return cv2.warpPerspective(img_board, H, (size[1], size[0]), borderValue=128)


def test_board_pose_is_recovered_from_an_image():
    cfg = TGT["hand_board"]
    T = make_transform(rot([1, 0.3, 0], 25), [0.02, -0.03, 0.9])     # board 0.9 m away, tilted 25 deg
    gray = render(board_image(cfg, 800), 800, T, K)          # ~ the final scale: no aliasing
    T_est, n, err = detect_board(gray, cfg, K)
    assert T_est is not None and n >= 8 and err < 1.0
    dp, dr = pose_error(T_est, T)
    assert dp < 3.0 and dr < 0.5


def test_apriltag_pose_is_recovered():
    a = TGT["apriltags"]
    img = apriltag_image(a["dictionary"], 0, px=160, white_border_frac=0.25)
    px_per_m = 160 / a["size_m"]
    T_center = make_transform(rot([1, 0, 0], 20), [0.05, 0.0, 0.8])
    # the rendered texture's origin is its top-left corner: the tag center sits (0.5 + border) * size inside
    off = (0.5 + 0.25) * a["size_m"]
    gray = render(img, px_per_m, T_center @ make_transform(np.eye(3), [-off, -off, 0]), K)
    tags = detect_tags(gray, a["dictionary"], a["size_m"], K)
    assert 0 in tags
    dp, dr = pose_error(tags[0], T_center)
    assert dp < 5.0 and dr < 2.0


def _poses(rng, n):
    out = []
    for _ in range(n):
        R = rot(rng.normal(size=3), rng.uniform(15, 40)) @ rot([1, 0, 0], 180)
        out.append(make_transform(R, [rng.uniform(0.35, 0.6), rng.uniform(-0.2, 0.2), rng.uniform(0.3, 0.5)]))
    return out


def _noisy(T, rng, mm=0.5, deg=0.1):
    return make_transform(rot(rng.normal(size=3), rng.normal(0, deg)) @ T[:3, :3],
                          T[:3, 3] + rng.normal(0, mm / 1000, 3))


def test_eye_in_hand_recovers_the_wrist_camera():
    rng = np.random.default_rng(0)
    X = make_transform(rot([0, 1, 0], -25) @ rot([0, 0, 1], 90), [0.07, 0.0, 0.035])       # hand -> camera
    T_base_board = make_transform(np.eye(3), [0.5, -0.1, 0.0])
    A = _poses(rng, 15)
    B = [_noisy(np.linalg.inv(X) @ np.linalg.inv(a) @ T_base_board, rng) for a in A]
    X_est = solve_hand_eye(A, B, "eye_in_hand")
    dp, dr = pose_error(X_est, X)
    assert dp < 2.0 and dr < 0.2
    r = residuals(A, B, X_est, "eye_in_hand")
    assert r["pos_mm_rms"] < 3.0
    assert pose_error(r["constant"], T_base_board)[0] < 2.0


def test_eye_to_hand_recovers_the_fixed_camera():
    rng = np.random.default_rng(1)
    eye = np.array([1.10, 0.50, 0.65])
    X = make_transform(look_at_optical(eye, [0.5, 0.05, 0.0]), eye)                         # base -> camera
    T_hand_board = mount_T(TGT["hand_board"])
    A = _poses(rng, 15)
    B = [_noisy(np.linalg.inv(X) @ a @ T_hand_board, rng) for a in A]
    X_est = solve_hand_eye(A, B, "eye_to_hand")
    dp, dr = pose_error(X_est, X)
    assert dp < 3.0 and dr < 0.2
    assert pose_error(residuals(A, B, X_est, "eye_to_hand")["constant"], T_hand_board)[0] < 3.0


def test_mount_and_board_geometry():
    cfg = TGT["hand_board"]
    T = mount_T(cfg) @ board_center_T(cfg)
    assert np.allclose(T[:3, 3], cfg["mount"]["center"])
    assert np.allclose(T[:3, 2], np.cross(cfg["mount"]["x_axis"], cfg["mount"]["y_axis"]))
    assert board_image(cfg, 1000).shape == (round(board_size_m(cfg)[1] * 1000), round(board_size_m(cfg)[0] * 1000))


def test_eye_to_hand_poses_face_the_camera_and_vary():
    from pnp_core.calibration import eye_to_hand_poses, rotation_spread_deg
    cfg = TGT["hand_board"]
    cam = np.array([1.1, 0.5, 0.65])
    rng = np.random.default_rng(0)
    poses = eye_to_hand_poses(cam, mount_T(cfg), cfg, 15, rng, [0.5, 0.05, 0.35], [0.08, 0.12, 0.08])
    for T in poses:
        Tb = T @ mount_T(cfg) @ board_center_T(cfg)
        to_cam = (cam - Tb[:3, 3]) / np.linalg.norm(cam - Tb[:3, 3])
        # printed face (-z) toward the camera, within the tilt budget (25 + 25 deg) + margin
        assert -Tb[:3, 2] @ to_cam > np.cos(np.radians(40))
        assert T[2, 2] < -0.3                  # hand z still points (roughly) down: reachable, fingers below
    assert rotation_spread_deg(poses) > 30


def test_eye_in_hand_poses_look_at_the_board():
    from pnp_core.calibration import eye_in_hand_poses, rotation_spread_deg
    from pnp_core.object_grasp import top_down_R
    from pnp_core.wrist import T_parent_optical
    wcfg = load_yaml("config/camera_d405.yaml")
    T_ho = T_parent_optical(wcfg)
    board_c = make_transform(np.diag([1.0, -1.0, -1.0]), [0.55, -0.10, 0.0])
    poses = eye_in_hand_poses(board_c, T_ho, top_down_R(0.0), 15, np.random.default_rng(1))
    for T in poses:
        To = T @ T_ho
        v = board_c[:3, 3] - To[:3, 3]
        d = np.linalg.norm(v)
        assert 0.2 < d < 0.4
        assert To[:3, 2] @ v / d > np.cos(np.radians(8))          # optical axis at the board center (+- jitter)
    assert rotation_spread_deg(poses) > 30


def test_refinement_beats_closed_form_on_pixel_noise():
    """Far board (0.7-0.9 m), few corners, 0.3 px noise: PnP poses are weak in tilt; refining in pixels fixes it."""
    from pnp_core.calibration import charuco_board, refine_hand_eye
    rng = np.random.default_rng(3)
    cfg = TGT["hand_board"]
    eye = np.array([1.10, 0.50, 0.65])
    X = make_transform(look_at_optical(eye, [0.5, 0.05, 0.0]), eye)
    C = mount_T(cfg)
    obj_all = charuco_board(cfg).getChessboardCorners().astype(float)
    A, B, obs = [], [], []
    for a in _poses(rng, 14):
        a = make_transform(a[:3, :3], a[:3, 3] + [0.05, 0.1, -0.1])
        T_cb = np.linalg.inv(X) @ a @ C
        obj = obj_all[rng.permutation(len(obj_all))[:8]]                 # a partial view: 8 corners
        img, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_cb[:3, :3])[0], T_cb[:3, 3], K, np.zeros(5))
        img = img.reshape(-1, 2) + rng.normal(0, 0.3, (len(obj), 2))
        ok, rv, tv = cv2.solvePnP(obj, img, K, np.zeros(5))
        A.append(a)
        B.append(make_transform(cv2.Rodrigues(rv)[0], tv.ravel()))
        obs.append((obj, img))
    X0 = solve_hand_eye(A, B, "eye_to_hand")
    C0 = residuals(A, B, X0, "eye_to_hand")["constant"]
    Xr, Cr, rms, per_view = refine_hand_eye(A, obs, X0, C0, "eye_to_hand", K)
    e0, er = pose_error(X0, X), pose_error(Xr, X)
    assert rms < 0.5 and len(per_view) == 14
    assert er[0] < 1.5 and er[1] < 0.1
    assert er[0] <= e0[0] + 0.1


def test_tag_corners_match_the_detector_and_project_back():
    from pnp_core.calibration import drift_state, project, tag_corners
    a = TGT["apriltags"]
    img = apriltag_image(a["dictionary"], 0, px=160, white_border_frac=0.25)
    px_per_m = 160 / a["size_m"]
    T_center = make_transform(rot([1, 0.2, 0], 25), [0.03, -0.02, 0.7])
    off = (0.5 + 0.25) * a["size_m"]
    gray = render(img, px_per_m, T_center @ make_transform(np.eye(3), [-off, -off, 0]), K)
    T, corners = detect_tags(gray, a["dictionary"], a["size_m"], K, with_corners=True)[0]
    pred = project(T_center, tag_corners(a["size_m"]), K)
    assert np.abs(pred - corners).max() < 1.0               # same order, same convention
    assert drift_state(0.5, 1.5, 3.0) == "OK" and drift_state(2.0, 1.5, 3.0) == "WARN" and drift_state(5, 1.5, 3) == "DRIFT"
