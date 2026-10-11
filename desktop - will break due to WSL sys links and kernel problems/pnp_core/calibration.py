"""Camera calibration math: ChArUco / AprilTag detection, hand-eye solving, residuals (docs/learning/16_calibration.md).

What this file teaches: where is a camera, relative to the robot? Show it a known pattern from many robot poses.
For each pose i we know A_i = T_base_hand (robot kinematics) and measure B_i = T_cam_board (the pattern's pose in
the image, solvePnP on the ChArUco corners). One transform is constant through all poses, and that is the answer:

  eye-in-hand (wrist camera, board on the table):  T_base_board = A_i  @ X @ B_i     X = T_hand_cam  (unknown)
  eye-to-hand (fixed camera, board in the gripper): T_hand_board = A_i^-1 @ X @ B_i  X = T_base_cam  (unknown)

Both are the classic AX = XB problem (cv2.calibrateHandEye: Tsai, Park, Horaud, Andreff, Daniilidis). The
residual is how much the "constant" side really varies over the poses: a few mm / tenths of a degree is good.
Needs varied ROTATIONS between poses (pure translations leave X unobservable).
"""

from __future__ import annotations

import numpy as np

try:                    # the geometry helpers (mount_T, board sizes) are also used inside Isaac Sim, which has no cv2
    import cv2
except ImportError:     # pragma: no cover
    cv2 = None

from .geometry import make_transform


# --- targets ---------------------------------------------------------------------------------------------------
def charuco_board(cfg: dict):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    n = (cfg["squares_x"] * cfg["squares_y"]) // 2
    ids = np.arange(cfg["first_id"], cfg["first_id"] + n, dtype=np.int32)
    return cv2.aruco.CharucoBoard((cfg["squares_x"], cfg["squares_y"]), cfg["square_m"], cfg["marker_m"], d, ids)


def board_size_m(cfg: dict):
    return cfg["squares_x"] * cfg["square_m"], cfg["squares_y"] * cfg["square_m"]


def board_image(cfg: dict, px_per_m: float = 2000.0) -> np.ndarray:
    """The printable board, exactly squares_x*square_m wide at px_per_m (grey image, no margin)."""
    w, h = board_size_m(cfg)
    return charuco_board(cfg).generateImage((int(round(w * px_per_m)), int(round(h * px_per_m))), marginSize=0,
                                            borderBits=1)


def apriltag_image(dictionary: str, tag_id: int, px: int = 800, white_border_frac: float = 0.25) -> np.ndarray:
    """A tag image: the black square fills `px`, plus a white quiet zone around it."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    tag = cv2.aruco.generateImageMarker(d, tag_id, px)
    b = int(px * white_border_frac)
    return cv2.copyMakeBorder(tag, b, b, b, b, cv2.BORDER_CONSTANT, value=255)


def board_center_T(cfg: dict) -> np.ndarray:
    """Board CORNER frame (OpenCV's) -> center: T_corner_center (the center frame has the same axes)."""
    w, h = board_size_m(cfg)
    return make_transform(np.eye(3), [w / 2, h / 2, 0.0])


def mount_T(cfg: dict) -> np.ndarray:
    """T_parent_board(corner frame) from a `mount` {center, x_axis, y_axis} (z = x cross y, into the board)."""
    m = cfg["mount"]
    x, y = np.asarray(m["x_axis"], float), np.asarray(m["y_axis"], float)
    R = np.column_stack([x, y, np.cross(x, y)])
    return make_transform(R, m["center"]) @ np.linalg.inv(board_center_T(cfg))


# --- detection -------------------------------------------------------------------------------------------------
def image_to_gray(m) -> np.ndarray:
    """A sensor_msgs/Image (duck-typed: data, encoding, height, width, step) -> grey uint8 array."""
    ch = {"mono8": 1, "rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}[m.encoding]
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)
    code = {"rgb8": cv2.COLOR_RGB2GRAY, "bgr8": cv2.COLOR_BGR2GRAY, "rgba8": cv2.COLOR_RGBA2GRAY,
            "bgra8": cv2.COLOR_BGRA2GRAY}.get(m.encoding)
    return a[:, :, 0].copy() if code is None else cv2.cvtColor(a, code)



def detect_board(gray: np.ndarray, cfg: dict, K: np.ndarray, D=None, min_corners: int = 6, return_points=False):
    """-> (T_cam_board 4x4 | None, n_corners, reprojection error [px]) [+ (board points Nx3, image points Nx2) with
    return_points]. Board frame = OpenCV's (corner origin)."""
    board = charuco_board(cfg)
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(gray)
    fail = (None, 0 if ids is None else len(ids), float("nan")) + ((None,) if return_points else ())
    if ids is None or len(ids) < min_corners:
        return fail
    obj, img = board.matchImagePoints(corners, ids)
    D = np.zeros(5) if D is None else np.asarray(D, float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return fail
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
    out = (make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel()), len(ids), err)
    return out + ((obj.reshape(-1, 3), img.reshape(-1, 2)),) if return_points else out


def tag_corners(size_m: float) -> np.ndarray:
    """The 4 corners in the tag frame (x right, y down, z into the tag), in the detector's order TL, TR, BR, BL."""
    s = size_m / 2
    return np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], float)


def detect_tags(gray: np.ndarray, dictionary: str, size_m: float, K: np.ndarray, D=None,
                with_corners: bool = False) -> dict:
    """-> {id: T_cam_tag} (tag frame: center, x right, y down, z into the tag, OpenCV convention), or with_corners
    {id: (T_cam_tag, 4x2 detected corner pixels in tag_corners order)}."""
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX     # sub-pixel corners: single-tag poses need it
    corners, ids, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
    out = {}
    if ids is None:
        return out
    s = size_m / 2
    # the corner order SOLVEPNP_IPPE_SQUARE requires (TL, TR, BR, BL with y UP; detector corners come TL, TR, BR, BL)
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float32)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])            # -> y down, z into the tag (same convention as the boards)
    D = np.zeros(5) if D is None else np.asarray(D, float)
    for c, i in zip(corners, ids.ravel()):
        ok, rvec, tvec = cv2.solvePnP(obj, c.reshape(-1, 2), K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            T = make_transform(cv2.Rodrigues(rvec)[0], tvec.ravel()) @ flip
            out[int(i)] = (T, c.reshape(-1, 2)) if with_corners else T
    return out


def project(T_cam_X: np.ndarray, pts_X: np.ndarray, K: np.ndarray, D=None) -> np.ndarray:
    """Pixels of points given in frame X, seen by a camera at T_cam_X."""
    D = np.zeros(5) if D is None else np.asarray(D, float)
    uv, _ = cv2.projectPoints(np.asarray(pts_X, float), cv2.Rodrigues(T_cam_X[:3, :3])[0], T_cam_X[:3, 3], K, D)
    return uv.reshape(-1, 2)


def drift_state(px: float, warn_px: float, drift_px: float) -> str:
    return "OK" if px < warn_px else ("WARN" if px < drift_px else "DRIFT")


# --- hand-eye ----------------------------------------------------------------------------------------------------
def solve_hand_eye(T_base_hand: list, T_cam_board: list, mode: str, method=None) -> np.ndarray:
    """mode 'eye_in_hand' -> T_hand_cam; 'eye_to_hand' -> T_base_cam (pass base->hand inverted, the standard trick)."""
    method = cv2.CALIB_HAND_EYE_PARK if method is None else method
    if mode == "eye_in_hand":
        A = T_base_hand
    elif mode == "eye_to_hand":
        A = [np.linalg.inv(T) for T in T_base_hand]
    else:
        raise ValueError(mode)
    R, t = cv2.calibrateHandEye([T[:3, :3] for T in A], [T[:3, 3] for T in A],
                                [T[:3, :3] for T in T_cam_board], [T[:3, 3] for T in T_cam_board], method=method)
    return make_transform(R, t.ravel())


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def constant_side(T_base_hand, T_cam_board, X, mode):
    """The transform that should be the SAME for every pose: T_base_board (eye-in-hand) or T_hand_board
    (eye-to-hand). -> list of 4x4."""
    if mode == "eye_in_hand":
        return [A @ X @ B for A, B in zip(T_base_hand, T_cam_board)]
    return [np.linalg.inv(A) @ X @ B for A, B in zip(T_base_hand, T_cam_board)]


def mean_transform(Ts: list) -> np.ndarray:
    """Average pose (rotation: projected mean of matrices, fine for small spreads)."""
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(np.mean([T[:3, :3] for T in Ts], axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return make_transform(R, t)


def residuals(T_base_hand, T_cam_board, X, mode) -> dict:
    """How constant the 'constant' side really is: per pose position [mm] / rotation [deg] from its mean."""
    C = constant_side(T_base_hand, T_cam_board, X, mode)
    M = mean_transform(C)
    pos = [float(np.linalg.norm(T[:3, 3] - M[:3, 3]) * 1000) for T in C]
    rot = [rotation_angle_deg(M[:3, :3].T @ T[:3, :3]) for T in C]
    return {"pos_mm": pos, "rot_deg": rot, "pos_mm_rms": float(np.sqrt(np.mean(np.square(pos)))),
            "rot_deg_rms": float(np.sqrt(np.mean(np.square(rot)))), "constant": M}


def _view_transform(A, X, C, mode):
    """T_cam_board of one view, from the robot pose A = T_base_hand, the camera X and the constant side C."""
    if mode == "eye_in_hand":                   # T_base_board = A X B  ->  B = X^-1 A^-1 C
        return np.linalg.inv(X) @ np.linalg.inv(A) @ C
    return np.linalg.inv(X) @ A @ C             # T_hand_board = A^-1 X B  ->  B = X^-1 A C


def refine_hand_eye(T_base_hand, observations, X0, C0, mode, K, D=None):
    """Joint refinement (a small bundle adjustment): find X (the camera) and C (the constant side: board in the hand
    / on the table) that minimize the PIXEL error of every detected board corner in every view.

    The closed-form solvers treat each view's solvePnP pose as exact and mix millimeters with degrees in their own
    way; a far-away board's PnP pose is good across the image but weak in depth and tilt. Refining in pixels weighs
    each measurement by what the camera really measured. observations: per view (board points Nx3, pixels Nx2).
    -> (X, C, rms_px, per-view rms_px)"""
    from scipy.optimize import least_squares
    D = np.zeros(5) if D is None else np.asarray(D, float)

    def pack(T):
        return np.concatenate([cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]])

    def unpack(p):
        return make_transform(cv2.Rodrigues(p[:3])[0], p[3:6])

    def res(p):
        X, C = unpack(p[:6]), unpack(p[6:])
        out = []
        for A, (obj, img) in zip(T_base_hand, observations):
            B = _view_transform(A, X, C, mode)
            proj, _ = cv2.projectPoints(np.asarray(obj, float), cv2.Rodrigues(B[:3, :3])[0], B[:3, 3], K, D)
            out.append((proj.reshape(-1, 2) - np.asarray(img, float)).ravel())
        return np.concatenate(out)

    sol = least_squares(res, np.concatenate([pack(X0), pack(C0)]), method="lm", x_scale="jac")
    X, C = unpack(sol.x[:6]), unpack(sol.x[6:])
    r = res(sol.x).reshape(-1, 2)
    sizes = np.cumsum([0] + [len(o[0]) for o in observations])
    per_view = [float(np.sqrt(np.mean(np.sum(r[a:b] ** 2, axis=1)))) for a, b in zip(sizes[:-1], sizes[1:])]
    return X, C, float(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))), per_view


def axis_angle_R(axis, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


def rotation_spread_deg(Ts: list) -> float:
    """Largest rotation between any two of the poses: hand-eye needs this well above ~20 deg (and about different
    axes) or the rotation part of X is poorly observable."""
    return max((rotation_angle_deg(A[:3, :3].T @ B[:3, :3]) for i, A in enumerate(Ts) for B in Ts[i + 1:]), default=0.0)


# --- pose planning ----------------------------------------------------------------------------------------------
def eye_to_hand_poses(cam_pos, T_hand_board: np.ndarray, board_cfg: dict, n: int, rng, center, spread,
                      tilt_deg: float = 25.0, roll_deg: float = 30.0, y_ref: str = "down") -> list:
    """Hand poses (T_base_hand) that show the gripper-held board to a FIXED camera at cam_pos: board centers in a box
    (center +- spread), the printed face turned toward the camera, then randomly tilted (about the board's x and y)
    and rolled (about its normal) so the rotations differ. The board's y axis starts pointing
      down    (a camera beside the robot: the FR3 holds the board hanging below its hand), or
      radial  away from the robot base (a camera ABOVE: the SO-101 holds it out flat; "down" would be along z)."""
    T_hand_c = T_hand_board @ board_center_T(board_cfg)                       # hand -> board CENTER
    out = []
    for _ in range(n):
        p = np.asarray(center, float) + rng.uniform(-1, 1, 3) * np.asarray(spread, float)
        z = p - np.asarray(cam_pos, float)                                     # board z points AWAY from the camera
        z /= np.linalg.norm(z)
        ref = np.array([p[0], p[1], 0.0]) / max(1e-9, np.hypot(p[0], p[1])) if y_ref == "radial" \
            else np.array([0.0, 0.0, -1.0])
        y = ref - z * (ref @ z)                                                # the reference, in the board plane
        y /= np.linalg.norm(y)
        R = np.column_stack([np.cross(y, z), y, z])
        R = R @ axis_angle_R([1, 0, 0], np.radians(rng.uniform(-tilt_deg, tilt_deg)))               @ axis_angle_R([0, 1, 0], np.radians(rng.uniform(-tilt_deg, tilt_deg)))               @ axis_angle_R([0, 0, 1], np.radians(rng.uniform(-roll_deg, roll_deg)))
        out.append(make_transform(R, p) @ np.linalg.inv(T_hand_c))
    return out


def planar_arm_hand_poses(n: int, rng, radius=(0.12, 0.20), height=(0.12, 0.20), pan_rad=(-0.5, 0.5),
                          pitch_deg=(-25.0, 25.0), roll_deg=(-30.0, 30.0)) -> list:
    """Hand poses for a 5-joint arm (SO-101) holding a board out in front, face up, under an overhead camera.

    Such an arm can only point its hand WITHIN the vertical plane through its base axis (only the base pan turns that
    plane), so a free yaw like eye_to_hand_poses' is unreachable. Here: hand origin at radius / height / pan, the hand
    z (out of the jaws) along the pan direction pitched by pitch_deg (positive = down), the hand x pointing down
    (= a board held in the jaws, printed face toward -x, faces UP), rolled about z by roll_deg. Rotations about three
    different axes (pan, pitch, roll): what hand-eye calibration needs."""
    out = []
    for _ in range(n):
        phi = rng.uniform(*pan_rad)
        th = np.radians(rng.uniform(*pitch_deg))
        r, h = rng.uniform(*radius), rng.uniform(*height)
        z = np.array([np.cos(phi) * np.cos(th), np.sin(phi) * np.cos(th), -np.sin(th)])
        down = np.array([0.0, 0.0, -1.0])
        x = down - z * (down @ z)
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(z, x), z]) @ axis_angle_R([0, 0, 1], np.radians(rng.uniform(*roll_deg)))
        out.append(make_transform(R, [r * np.cos(phi), r * np.sin(phi), h]))
    return out


def joint_space_board_views(fk_hand, lower, upper, n: int, rng, T_base_cam, K, size, T_hand_board, board_cfg,
                            dist=(0.2, 0.45), max_view_angle_deg=45.0, min_height=0.08, ranges=None, tries=200000,
                            min_visible=1.0, min_corner_height=0.02):
    """Arm configurations (joint vectors) that show a hand-held board to a FIXED camera: sampled in JOINT space, so
    every one is reachable by construction (any arm, incl. 5-joint ones whose hand can't take arbitrary poses).
    Kept if forward kinematics puts the board's printed face toward the camera (within max_view_angle_deg), at least
    `min_visible` of its inner corners in the image (1.0 = all of it; ChArUco works with part of the board, which a
    camera that isn't straight above the arm needs), `dist` from the camera, its center above min_height and every
    outer corner above min_corner_height (a tilted board's edge reaches down: it touched the desk).
    fk_hand(q) -> T_base_hand. ranges: optional per-joint (lo, hi) to sample in (default: the joint limits). -> [q]"""
    lo, hi = (np.asarray(lower, float), np.asarray(upper, float)) if ranges is None else \
        (np.array([r[0] for r in ranges], float), np.array([r[1] for r in ranges], float))
    T_cam_base = np.linalg.inv(T_base_cam)
    T_hand_c = T_hand_board @ board_center_T(board_cfg)
    w, h = board_size_m(board_cfg)
    outer = np.array([[sx * w / 2, sy * h / 2, 0, 1] for sx in (-1, 1) for sy in (-1, 1)])
    corners = outer
    if min_visible < 1.0:                              # the board's inner (ChArUco) corners, center frame
        sq, nx, ny = board_cfg["square_m"], board_cfg["squares_x"], board_cfg["squares_y"]
        gx, gy = np.meshgrid((np.arange(1, nx) - nx / 2) * sq, (np.arange(1, ny) - ny / 2) * sq)
        corners = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size), np.ones(gx.size)])
    out = []
    for _ in range(tries):
        q = rng.uniform(lo, hi)
        Tb = fk_hand(q) @ T_hand_c
        if Tb[2, 3] < min_height or np.min(outer @ Tb[2]) < min_corner_height:
            continue
        to_cam = T_base_cam[:3, 3] - Tb[:3, 3]
        d = float(np.linalg.norm(to_cam))
        if not dist[0] <= d <= dist[1] or (-Tb[:3, 2]) @ to_cam / d < np.cos(np.radians(max_view_angle_deg)):
            continue
        P = corners @ (T_cam_base @ Tb).T
        if np.any(P[:, 2] < 0.05):
            continue
        uv = (P[:, :3] @ np.asarray(K).T)[:, :2] / P[:, 2:3]
        if np.mean(np.all((uv > 25) & (uv < np.asarray(size) - 25), axis=1)) >= min_visible - 1e-9:
            out.append(q)
            if len(out) >= n:
                break
    return out


def joint_space_wrist_views(fk_hand, lower, upper, n: int, rng, T_hand_opt, T_base_board_c, K, size, board_cfg,
                            dist=(0.15, 0.24), max_view_angle_deg=35.0, min_cam_height=0.08, min_tcp_height=0.04,
                            fk_tcp=None, ranges=None, region_px=None, tries=200000):
    """Arm configurations that show a board lying on the table to the WRIST camera (eye-in-hand), sampled in JOINT
    space like joint_space_board_views (reachable by construction: a 5-joint arm can't look from arbitrary poses).
    Kept if the camera is `dist` from the board's center, sees its printed face (from above, within
    max_view_angle_deg of its normal), all of it in the image, the camera above min_cam_height and the TCP (fk_tcp)
    above min_tcp_height. fk_hand(q) -> T_base_hand; T_hand_opt: the (nominal) camera on the hand.
    region_px: (u0, v0, u1, v1) the board must lie in (default: the image minus 25 px), e.g. above the jaws. -> [q]"""
    lo, hi = (np.asarray(lower, float), np.asarray(upper, float)) if ranges is None else         (np.array([r[0] for r in ranges], float), np.array([r[1] for r in ranges], float))
    c, up = T_base_board_c[:3, 3], -T_base_board_c[:3, 2]          # the printed face looks along -z (OpenCV board)
    w, h = board_size_m(board_cfg)
    corners = np.array([[sx * w / 2, sy * h / 2, 0, 1] for sx in (-1, 1) for sy in (-1, 1)]) @ T_base_board_c.T
    r0, r1 = (np.array([25.0, 25.0]), np.asarray(size, float) - 25) if region_px is None else         (np.asarray(region_px[:2], float), np.asarray(region_px[2:], float))
    out = []
    for _ in range(tries):
        q = rng.uniform(lo, hi)
        T_cam = fk_hand(q) @ T_hand_opt
        p = T_cam[:3, 3]
        d = float(np.linalg.norm(p - c))
        if p[2] < min_cam_height or not dist[0] <= d <= dist[1] or up @ (p - c) / d < np.cos(np.radians(max_view_angle_deg)):
            continue
        P = (corners[:, :3] - p) @ T_cam[:3, :3]                    # corners in the camera frame
        if np.any(P[:, 2] < 0.05):
            continue
        uv = (P @ np.asarray(K).T)[:, :2] / P[:, 2:3]
        if not np.all((uv > r0) & (uv < r1)):
            continue
        if fk_tcp is not None and fk_tcp(q)[2, 3] < min_tcp_height:
            continue
        out.append(q)
        if len(out) >= n:
            break
    return out


def eye_in_hand_poses(T_base_board_c: np.ndarray, T_hand_opt: np.ndarray, R_hand_ref: np.ndarray, n: int, rng,
                      dist=(0.24, 0.36), max_tilt_deg: float = 30.0, roll_deg: float = 40.0,
                      look_jitter_m: float = 0.02) -> list:
    """Hand poses that put the WRIST camera on a cap above a board lying at T_base_board_c (its center): distance
    `dist`, view tilted up to max_tilt_deg from straight down, looking at the center (+- jitter), and rolled about
    the view axis within +- roll_deg of the roll it has when the hand is at R_hand_ref (a comfortable top-down hand
    orientation), so the arm isn't asked to twist its wrist around."""
    c = T_base_board_c[:3, 3]
    x_ref = (R_hand_ref @ T_hand_opt[:3, :3])[:, 0]                            # camera x with the hand at R_hand_ref
    out = []
    for _ in range(n):
        tilt, az = np.radians(rng.uniform(0, max_tilt_deg)), rng.uniform(0, 2 * np.pi)
        cam = c + rng.uniform(*dist) * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
        z = c + rng.uniform(-1, 1, 3) * [look_jitter_m, look_jitter_m, 0] - cam
        z /= np.linalg.norm(z)
        x = x_ref - z * (x_ref @ z)
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(z, x), z]) @ axis_angle_R([0, 0, 1], np.radians(rng.uniform(-roll_deg, roll_deg)))
        out.append(make_transform(R, cam) @ np.linalg.inv(T_hand_opt))
    return out


def pose_error(T_a: np.ndarray, T_b: np.ndarray):
    """-> (position error [mm], rotation error [deg]) between two poses."""
    return (float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000), rotation_angle_deg(T_a[:3, :3].T @ T_b[:3, :3]))


def flat_tag_T(xy, yaw_deg: float, z: float) -> np.ndarray:
    """A tag lying printed side up on the board at (x, y, z), base frame: x along yaw, y "down" the print, z into it."""
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    return make_transform(np.array([[c, s, 0], [s, -c, 0], [0, 0, -1.0]]), [xy[0], xy[1], z])


def tag_layout(apriltags: dict, survey: dict | None = None) -> dict:
    """id -> (x, y, yaw_deg) of each tag on the board: the SURVEYED position (survey: a <set>_tag_layout.yaml's
    `tags`, tools/calibrate.py locate --survey) where there is one, else the planned one (sim_xy, sim_yaw_deg)."""
    out = {int(t["id"]): (float(t["sim_xy"][0]), float(t["sim_xy"][1]), float(t.get("sim_yaw_deg", 0.0)))
           for t in apriltags["tags"]}
    for k, v in ((survey or {}).get("tags") or {}).items():
        out[int(k)] = (float(v["xy"][0]), float(v["xy"][1]), float(v["yaw_deg"]))
    return out


def survey_tags(obs: dict, size_m: float, K: np.ndarray, D, planned: dict, z: float, X0: np.ndarray,
                prior_mm: float = 15.0, prior_deg: float = 5.0, px_sigma: float = 0.3, scale_sigma: float = 0.002,
                outlier_mm: float = 4.0, _free=()):
    """Where the tags REALLY lie, and the camera, from ONE image of them (all on the board plane z).

    The tags were placed by hand (a tape measure: ~1-2 cm). Unknowns: the camera pose (6) + each seen tag's x, y, yaw
    on the board (3 each). Observations: every corner's pixel (8 per tag). The tags' positions RELATIVE to each other
    come out to well under a mm (corners are good to ~0.2 px); where the whole set lies relative to the robot base
    (x, y, yaw of the lot) the image can't tell -- that comes from the planned positions as a prior (each placed within
    ~prior_mm / prior_deg), i.e. the average placement error over the tags (~prior_mm / sqrt(n)). Nor its SCALE, really:
    the whole layout a little bigger + the camera a little farther fits the pixels almost as well, only the tags' size
    in the image argues (and a printed tag's apparent size is biased: blur, print scale). In the sim a free scale
    spread the exact layout by 2 % (6 mm at the far tags, the camera 12 mm off). A tape measure gets scale right to
    ~0.2 % (scale_sigma): the offsets' common expansion is held to that.

    obs: {id: 4x2 corner pixels (detect_tags order)}; planned: {id: (x, y, yaw_deg)}; X0: base -> optical (a start).
    The tags' APPARENT size is free too (one factor for all): their corners detect a bit outside the printed square
    (blur; on paper also toner spread): +2 % in the sim. With the size fixed, the layout + camera bent to explain it.

    One tag put far from its spot would drag that pinning (its 2 cm shared out as ~2 mm on every other tag): tags
    found > outlier_mm off are left out of it and the survey solved again (up to 3 passes).

    -> (X base -> optical, {id: (x, y, yaw_deg)}, reprojection rms px, apparent tag size m)."""
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    ids = sorted(i for i in obs if i in planned)
    D = np.zeros(5) if D is None else np.asarray(D, float)
    Xi0 = np.linalg.inv(X0)                                  # optical <- base
    p0 = np.concatenate([Rotation.from_matrix(Xi0[:3, :3]).as_rotvec(), Xi0[:3, 3], [0.0], np.zeros(3 * len(ids))])
    P = np.array([planned[i][:2] for i in ids], float)
    # the 4 whole-set modes as unit-weight rows over (dx, dy per tag [m], dyaw per tag [rad]): shift x, shift y, a turn
    # about the set's centre (positions AND yaws), a scale about it. Their components are held near 0 (gauge_sigma).
    n_ = len(ids)
    w = np.array([0.0 if i in _free else 1.0 for i in ids])       # the tags that define the set's placement
    c_ = (P - (w[:, None] * P).sum(axis=0) / w.sum()) * w[:, None]
    w2 = np.repeat(w, 2)
    rows = [np.concatenate([np.tile([1.0, 0.0], n_) * w2, np.zeros(n_)]),
            np.concatenate([np.tile([0.0, 1.0], n_) * w2, np.zeros(n_)]),
            np.concatenate([np.column_stack([-c_[:, 1], c_[:, 0]]).ravel(), w]),
            np.concatenate([c_.ravel(), np.zeros(n_)])]
    M = np.array([r_ / np.dot(r_, r_) for r_ in rows])      # row . offsets = that mode's amount (m, rad, rad, ratio)
    gauge_sigma = np.array([0.5e-3, 0.5e-3, 1e-3, scale_sigma])

    def unpack(p):
        Xi = make_transform(Rotation.from_rotvec(p[:3]).as_matrix(), p[3:6])
        lay = {i: (planned[i][0] + p[7 + 3 * k] / 1000, planned[i][1] + p[8 + 3 * k] / 1000,
                   planned[i][2] + p[9 + 3 * k]) for k, i in enumerate(ids)}
        return Xi, lay, size_m * (1 + p[6] / 100)              # p[6]: the apparent size, % from the printed

    def res(p):
        Xi, lay, sz = unpack(p)
        C = tag_corners(sz)
        r = []
        rvec = cv2.Rodrigues(Xi[:3, :3])[0]
        for i in ids:
            T = flat_tag_T(lay[i][:2], lay[i][2], z)
            uv, _ = cv2.projectPoints(C @ T[:3, :3].T + T[:3, 3], rvec, Xi[:3, 3], K, D)
            r += list(((uv.reshape(-1, 2) - obs[i]) / px_sigma).ravel())
        d = p[7:].reshape(-1, 3)
        r += list((d[:, :2] / prior_mm).ravel()) + list(d[:, 2] / prior_deg)
        # the set's whole-board motions (common shift x / y, turn, scale) are pinned to the tape plan (see above)
        r += list(M @ np.concatenate([d[:, :2].ravel() / 1000, np.radians(d[:, 2])]) / gauge_sigma)
        return np.array(r)

    sol = least_squares(res, p0, x_scale="jac")
    Xi, lay, sz = unpack(sol.x)
    n = 8 * len(ids)
    rms = float(np.sqrt(np.mean((sol.fun[:n] * px_sigma) ** 2) * 2))
    far = {i for i in ids if np.hypot(lay[i][0] - planned[i][0], lay[i][1] - planned[i][1]) * 1000 > outlier_mm}
    if far - set(_free) and len(_free) < 2 and len(ids) - len(far | set(_free)) >= 4:
        worst = max(far - set(_free), key=lambda i: np.hypot(lay[i][0] - planned[i][0], lay[i][1] - planned[i][1]))
        return survey_tags(obs, size_m, K, D, planned, z, np.linalg.inv(Xi), prior_mm, prior_deg, px_sigma,
                           scale_sigma, outlier_mm, tuple(_free) + (worst,))
    return np.linalg.inv(Xi), lay, rms, sz
