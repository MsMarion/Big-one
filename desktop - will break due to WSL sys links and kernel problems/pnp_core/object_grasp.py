"""Grasps and placements from a full 6-D object pose (FoundationPose) instead of "an upright can at (x, y)".

What this file teaches: once you know the object's pose T_base_obj (position AND orientation), grasping becomes
"where is the gripper relative to the OBJECT", not relative to the table:

    T_base_tcp = T_base_obj @ T_obj_tcp          (a grasp: gripper pose in the object's own frame)
    T_tcp_obj  = inv(T_base_tcp) @ T_base_obj    (the same thing seen from the gripper, once it holds the object)

T_tcp_obj is the key quantity: while the object is held it does not change, so
  * the object's pose while carried is T_base_tcp(now) @ T_tcp_obj (the pose node's ATTACHED state), and
  * to put the object down in a chosen pose you solve for the gripper: T_base_tcp = T_base_obj_goal @ inv(T_tcp_obj).

The object frame is the mesh frame FoundationPose estimates (sim/export_mesh.py: the USD asset's own frame, origin
at the bounding-box center). `usd_up_axis` in config/objects.yaml says which mesh axis is "up" when the object stands.

Grasp families:
  cap   (cylinder, e.g. the soup can) the gripper comes down along the can's axis and closes on the side wall just
        below the top rim: the upright can (also an upside-down one: whichever end points up)
  side  (cylinder) the can lies on its side: come down from above, fingers close across the diameter, palm along
        the axis
  box   (e.g. the cracker box) resting on any face: fingers close across a horizontal side that fits the open hand
        (with margin), `depth_below_top_m` below the top face, palm along the other horizontal side
Every grasp is "snapped" to an exactly vertical approach: the robot always comes straight down (Pilz LIN), and a
few degrees of real or estimated tilt are absorbed by the fingers. Orientations between the two families
(tilted more than `max_tilt` from both) get no grasp: the caller reports why instead of trying something risky.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import make_transform, matrix_to_quat, rot_z, wrap_angle
from .grasp import R_DOWN, GraspPlan, Pose

AXES = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1), "-X": (-1, 0, 0), "-Y": (0, -1, 0), "-Z": (0, 0, -1)}


def up_axis(obj_cfg: dict) -> np.ndarray:
    """The object's "up" direction in its own (mesh) frame."""
    return np.array(AXES[obj_cfg.get("usd_up_axis", "Z")], float)


def rotation_between(a, b) -> np.ndarray:
    """Smallest rotation R with R @ a = b (unit vectors)."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:                                   # opposite: 180 deg about any axis perpendicular to a
        p = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def upright_pose(obj_cfg: dict, xy, z_center: float, yaw: float = 0.0) -> np.ndarray:
    """T_base_obj of the object standing on its base at (x, y), its center at z_center, turned `yaw` about z.
    For the Y-up YCB assets this is the +90 deg rotation about x that the sim uses to stand them up."""
    R = rot_z(yaw) @ rotation_between(up_axis(obj_cfg), [0, 0, 1])
    return make_transform(R, [xy[0], xy[1], z_center])


def support_height(R_base_obj: np.ndarray, obj_cfg: dict) -> float:
    """Height of the object's center above the surface it rests on, in this orientation (lowest point = 0).
    Cylinder: the rim's lowest point; box: the projection of its half-extents on the vertical."""
    dims = obj_cfg["dims_m"]                  # [x, y, z] of the UPRIGHT object, z = height
    u = R_base_obj @ up_axis(obj_cfg)
    if obj_cfg["shape"] == "cylinder":
        uz = min(1.0, abs(float(u[2])))
        return dims[0] / 2 * np.sqrt(1 - uz * uz) + dims[2] / 2 * uz
    R_up = rotation_between(up_axis(obj_cfg), [0, 0, 1])          # object frame -> upright frame
    half_obj = np.abs(R_up.T) @ (np.asarray(dims) / 2)            # half-extents along the object's own axes
    return float(np.abs(R_base_obj[2, :]) @ half_obj)


def tilt_deg(R_base_obj: np.ndarray, obj_cfg: dict) -> float:
    """How far the object's up axis is from vertical [deg] (0 = standing, 90 = lying, 180 = upside down)."""
    u = R_base_obj @ up_axis(obj_cfg)
    return float(np.degrees(np.arccos(np.clip(u[2], -1.0, 1.0))))


def object_yaw(R_base_obj: np.ndarray, obj_cfg: dict) -> float:
    """The object's heading about world z: yaw of its x axis once 'up' is mapped to z (0 for upright_pose(yaw=0))."""
    x = R_base_obj @ rotation_between(up_axis(obj_cfg), [0, 0, 1]).T @ [1.0, 0, 0]
    return float(np.arctan2(x[1], x[0]))


def place_tcp_yaws(T_tcp_obj: np.ndarray, obj_cfg: dict, object_goal_yaw: float, symmetric_yaws,
                   current_yaw: float = 0.0):
    """Gripper yaws to try at the goal: any (symmetric_yaws) for a rotationally symmetric object, else the yaw(s)
    that turn the HELD object to object_goal_yaw (object yaw = gripper yaw + a constant offset). With
    `yaw_period_deg: 180` (a box) the object may also end up turned by 180 deg: both, the smaller wrist turn first."""
    g = obj_cfg["grasp"]
    if g.get("symmetric_yaw", False):
        return list(symmetric_yaws)
    offset = object_yaw(top_down_R(0.0) @ T_tcp_obj[:3, :3], obj_cfg)
    n = max(1, int(round(360.0 / float(g.get("yaw_period_deg", 360)))))
    yaws = [wrap_angle(object_goal_yaw - offset + 2 * np.pi * m / n) for m in range(n)]
    return sorted(yaws, key=lambda y: abs(wrap_angle(y - current_yaw)))


def rotation_error_deg(R_a: np.ndarray, R_b: np.ndarray, obj_cfg: dict) -> float:
    """Angle between two orientations of this object, ignoring what its symmetry makes invisible: for a
    rotationally symmetric object (a can) only the tilt of its up axis counts (a spin about the axis is the same
    pose); otherwise the full rotation, modulo yaw_period_deg about the up axis (a box turned 180 deg is the same box)."""
    g = obj_cfg["grasp"]
    a = up_axis(obj_cfg)
    if g.get("symmetric_yaw", False):
        return float(np.degrees(np.arccos(np.clip((R_a @ a) @ (R_b @ a), -1.0, 1.0))))
    n = max(1, int(round(360.0 / float(g.get("yaw_period_deg", 360)))))
    best = 180.0
    for k in range(n):
        th = 2 * np.pi * k / n
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        S = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K           # rotation about the up axis (object frame)
        dR = R_a.T @ R_b @ S
        best = min(best, float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1.0, 1.0)))))
    return best


def pose_agrees(T_base_obj: np.ndarray, det_xy, det_yaw, obj_cfg: dict, max_m: float = 0.015,
                max_frac: float = 0.35, max_deg: float = 20.0):
    """Does a 6-D pose agree with the detector (its xy + footprint yaw)? -> (ok, reason).

    Position: within max_m, but never more than max_frac of the object's smaller footprint side: 15 mm is fine for
    a 66 mm can, but on a 15.8 mm brick a pose 13 mm off is on the neighbouring spot (the SO-101 cell planned a grasp
    from a stale pose like that). Yaw (box-shaped objects lying flat, not square): the pose's footprint long side vs
    the detector's (a rectangle's outline repeats every 180 deg) within max_deg."""
    dims = np.asarray(obj_cfg["dims_m"], float)
    d = float(np.hypot(T_base_obj[0, 3] - det_xy[0], T_base_obj[1, 3] - det_xy[1]))
    lim = min(max_m, max_frac * float(min(dims[0], dims[1])))
    if d > lim:
        return False, f"6-D pose is {d * 1000:.0f} mm from the detector's estimate (limit {lim * 1000:.0f} mm)"
    if (det_yaw is not None and obj_cfg.get("shape") == "box" and abs(dims[0] - dims[1]) > 0.002
            and tilt_deg(T_base_obj[:3, :3], obj_cfg) < 20.0):
        long_yaw = object_yaw(T_base_obj[:3, :3], obj_cfg) + (0.0 if dims[0] >= dims[1] else np.pi / 2)
        e = abs(np.degrees(wrap_angle(2 * (long_yaw - det_yaw)))) / 2          # modulo 180 deg
        if e > max_deg:
            return False, f"6-D pose's yaw is {e:.0f} deg from the detector's"
    return True, "agrees with the detector"


def top_down_R(yaw: float) -> np.ndarray:
    return rot_z(yaw) @ R_DOWN


def tcp_yaw_of(R_base_tcp: np.ndarray) -> float:
    """Yaw of a top-down TCP (3x3 rotation or 4x4 transform)."""
    Rz = np.asarray(R_base_tcp)[:3, :3] @ R_DOWN.T
    return float(np.arctan2(Rz[1, 0], Rz[0, 0]))


def _closest_first(yaws, current_yaw):
    """Each yaw and yaw+pi are the same grasp for a parallel gripper: keep the one closer to the wrist's yaw."""
    out = []
    for y in yaws:
        y = min((wrap_angle(y), wrap_angle(y + np.pi)), key=lambda c: abs(wrap_angle(c - current_yaw)))
        if all(abs(wrap_angle(y - o)) > 1e-6 for o in out):
            out.append(y)
    return out


@dataclass
class ObjectGrasp:
    kind: str                 # "cap" | "side"
    T_base_tcp: np.ndarray    # 4x4: the TCP pose at the grasp
    T_tcp_obj: np.ndarray     # 4x4: the object seen from the TCP (constant while held)
    width_m: float            # expected finger width when holding

    def plan(self, open_width_m: float, force_n: float, approach_height=0.10, lift_height=0.15) -> GraspPlan:
        """The pre-grasp / grasp / lift poses the state machine executes (straight down, straight up)."""
        p, q = self.T_base_tcp[:3, 3], matrix_to_quat(self.T_base_tcp[:3, :3])
        up = np.array([0, 0, 1.0])
        return GraspPlan(pregrasp=Pose(p + approach_height * up, q), grasp=Pose(p.copy(), q),
                         lift=Pose(p + lift_height * up, q), open_width_m=float(open_width_m),
                         expected_width_m=float(self.width_m), force_n=float(force_n))


def with_flips(grasps: list) -> list:
    """+ each box grasp with the hand turned 180 deg about its approach axis. For two symmetric fingers the same
    grasp, for ONE fixed jaw (SO-101) a different one: the fixed jaw on the other side of the object (away from an
    obstacle, or a wrist roll that also reaches the place). _box_grasps keeps only the one closer to the wrist."""
    F = np.diag([-1.0, -1.0, 1.0, 1.0])
    return list(grasps) + [ObjectGrasp(g.kind, g.T_base_tcp @ F, F @ g.T_tcp_obj, g.width_m)
                           for g in grasps if g.kind == "box"]


def shift_for_fixed_jaw(grasps: list, held_offset_y, approach_clearance: float = 0.0) -> list:
    """Grippers with ONE moving jaw (SO-101) push the object against the fixed jaw: a held object of width w sits
    at TCP y = held_offset_y(w), not at y = 0. Each grasp is moved so the fixed jaw comes down `approach_clearance`
    beside the object (flush, any detection error put the fixed jaw ON the object), and the closing jaw then slides
    the object that last bit against it. T_tcp_obj = where the object IS once held (y = held_offset_y), so placing
    stays exact. held_offset_y = GripperModel.held_offset_y."""
    out = []
    for g in grasps:
        off = float(held_offset_y(g.width_m))
        if abs(off) < 1e-6 and approach_clearance == 0.0:
            out.append(g)
            continue
        T_obj = g.T_base_tcp @ g.T_tcp_obj
        T_held = g.T_base_tcp @ make_transform(np.eye(3), [0.0, -off, 0.0])           # the TCP once the object is held
        T_approach = g.T_base_tcp @ make_transform(np.eye(3), [0.0, -(off + approach_clearance), 0.0])
        out.append(ObjectGrasp(g.kind, T_approach, np.linalg.inv(T_held) @ T_obj, g.width_m))
    return out


def object_extents(obj_cfg: dict) -> np.ndarray:
    """Size of the object along its OWN (mesh) axes x, y, z. (dims_m is given for the upright object.)"""
    R_up = rotation_between(up_axis(obj_cfg), [0, 0, 1])
    return np.abs(R_up.T) @ np.asarray(obj_cfg["dims_m"], float)


def _box_grasps(T_base_obj, obj_cfg, current_yaw, max_tilt_deg, gripper_max_width_m):
    """Box: whichever of its axes is (nearly) vertical, close the fingers across a HORIZONTAL side that fits the
    open hand with `finger_margin_m` to spare on each side; the palm (TCP x) lies along the other horizontal side."""
    g = obj_cfg["grasp"]
    margin, depth = float(g.get("finger_margin_m", 0.002)), float(g["depth_below_top_m"])
    R, c = T_base_obj[:3, :3], T_base_obj[:3, 3]
    ext = object_extents(obj_cfg)
    k = int(np.argmax(np.abs(R[2, :])))                       # the object axis closest to vertical
    lean = float(np.degrees(np.arccos(min(1.0, abs(R[2, k])))))
    if lean > max_tilt_deg:
        return [], f"box tilted {lean:.0f} deg from resting on a face: no safe top-down grasp"
    s = 1.0 if R[2, k] > 0 else -1.0
    p = c + s * R[:, k] * max(0.0, ext[k] / 2 - depth)       # `depth` below the top face (center if it's short)
    i, j = [a for a in range(3) if a != k]
    fits = [(f, palm) for f, palm in ((i, j), (j, i)) if ext[f] + 2 * margin <= gripper_max_width_m]
    if not fits:
        return [], (f"box on a {ext[i] * 1000:.0f} x {ext[j] * 1000:.0f} mm footprint: no side fits the "
                    f"{gripper_max_width_m * 1000:.0f} mm hand with {margin * 1000:.0f} mm margin")
    out = []
    for f, palm in sorted(fits, key=lambda fp: ext[fp[0]]):       # narrowest grip first (most clearance)
        # both hand orientations for this side (palm along +palm or -palm), the one closer to the current wrist
        # first: a 5-joint arm (SO-101) can't roll its wrist all the way round, so one of the two may be out of
        # reach; trying the flipped one before the WIDER side keeps the grip on the narrow side
        a = np.arctan2(R[1, palm], R[0, palm])
        for yaw in _closest_first([a, a + np.pi], current_yaw):
            T = make_transform(top_down_R(yaw), p)
            out.append(ObjectGrasp("box", T, np.linalg.inv(T) @ T_base_obj, float(ext[f])))
    return out, f"box grasp across {out[0].width_m * 1000:.1f} mm, tilt {lean:.1f} deg"


def grasps_from_pose(T_base_obj: np.ndarray, obj_cfg: dict, tcp_yaws, current_yaw: float = 0.0,
                     max_tilt_deg: float = 20.0, gripper_max_width_m: float = 0.08):
    """-> (list of ObjectGrasp best first, reason).

    tcp_yaws: preferred gripper yaws, best first (used where the object allows any yaw: the cap grasp of a
    rotationally symmetric object). The other cases get their yaws from the object's orientation."""
    if obj_cfg["shape"] == "box":
        return _box_grasps(T_base_obj, obj_cfg, current_yaw, max_tilt_deg, gripper_max_width_m)
    if obj_cfg["shape"] != "cylinder":
        return [], f"no grasp rules for shape '{obj_cfg['shape']}' yet"
    g = obj_cfg["grasp"]
    radius, height, depth = obj_cfg["dims_m"][0] / 2, obj_cfg["dims_m"][2], float(g["depth_below_top_m"])
    R, c = T_base_obj[:3, :3], T_base_obj[:3, 3]
    u = R @ up_axis(obj_cfg)                                  # the can's axis in the base frame
    tilt = tilt_deg(R, obj_cfg)
    lean = min(tilt, 180 - tilt)                              # 0 = axis vertical (standing or upside down)
    if lean <= max_tilt_deg:
        s = 1.0 if u[2] > 0 else -1.0                         # which end is up
        p = c + s * u * (height / 2 - depth)                  # on the axis, `depth` below the upper rim
        yaws = list(tcp_yaws)
        kind = "cap"
    elif lean >= 90 - max_tilt_deg:
        h = np.array([u[0], u[1], 0.0])
        h /= np.linalg.norm(h)
        p = c.copy()                                          # fingertips at the axis height, across the diameter
        yaws = _closest_first([np.arctan2(h[1], h[0])], current_yaw)   # TCP x along the axis
        kind = "side"
    else:
        return [], f"object tilted {tilt:.0f} deg: neither standing nor lying, no safe top-down grasp"
    out = []
    for y in yaws:
        T = make_transform(top_down_R(y), p)
        out.append(ObjectGrasp(kind, T, np.linalg.inv(T) @ T_base_obj, 2 * radius))
    return out, f"{kind} grasp, object tilt {tilt:.1f} deg"


def place_tcp_poses(T_tcp_obj: np.ndarray, obj_cfg: dict, goal_xy, table_z: float, clearance: float, tcp_yaws,
                    approach_height: float = 0.10):
    """For each gripper yaw (top-down): (yaw, pre-place Pose, place Pose, T_base_obj there) that sets the HELD
    object down centered on goal_xy, its lowest point `clearance` above the table.
    Uses the measured grasp offset: a can grasped 5 mm lower than planned is released 5 mm lower too."""
    out = []
    for y in tcp_yaws:
        R_tcp = top_down_R(y)
        R_obj = R_tcp @ T_tcp_obj[:3, :3]
        p_obj = np.array([goal_xy[0], goal_xy[1], table_z + clearance + support_height(R_obj, obj_cfg)])
        p_tcp = p_obj - R_tcp @ T_tcp_obj[:3, 3]
        q = matrix_to_quat(R_tcp)
        out.append((y, Pose(p_tcp + [0, 0, approach_height], q), Pose(p_tcp, q), make_transform(R_obj, p_obj)))
    return out


def nominal_grasp_offset(obj_cfg: dict) -> np.ndarray:
    """T_tcp_obj of the standard grasp of the object standing upright (before we have seen it: used to check
    that the goal is reachable before picking anything up)."""
    T = upright_pose(obj_cfg, [0.0, 0.0], obj_cfg["dims_m"][2] / 2)
    gs, _ = grasps_from_pose(T, obj_cfg, [0.0])
    return gs[0].T_tcp_obj
