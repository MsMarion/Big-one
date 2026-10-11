"""Turning objects over: resting poses, placing a held object on another face, and the graph of which faces can be
reached from which (docs/learning/19_reorient.md).

What this file teaches: an object on a table rests on one of a few stable FACES (a box: 6, a can: upright, upside
down, lying). A top-down grasp fixes the object in the hand (T_tcp_obj); to set it down on another face the hand has
to TURN with it:

    R_base_tcp = R_base_obj_goal @ R_tcp_obj.T          (the hand that holds the object turned like this)

Upright -> on its side is the hand turned 90 deg about the fingers' closing axis: it ends up horizontal. That works
only if the hand doesn't point UP (no placing from below) and nothing of the hand ends up below the table (a box laid
on the very face the fingers hold puts a finger under it). Faces that no single grasp + turn reaches can still be
reached in several moves (upright -> on its side -> upside down): a shortest path in the graph

    node = resting face        edge A -> B = some top-down grasp in A whose turn to B passes those two checks

The graph depends on the object AND the gripper (opening, hand size): the Franka hand can never lay the cracker box
flat (the only side that fits its 80 mm opening is the 72 mm one, and laying the box on it puts a finger under the
box), the SO-101 jaw with its 55 mm opening has other limits. Whether the ARM reaches the turned hand is checked
later, by MoveIt at the goal (pick_place_node) or by IK over a workspace grid (tools/grasp_study.py).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .geometry import rot_z
from .object_grasp import grasps_from_pose, rotation_between, support_height, up_axis

# turning the UPRIGHT frame (x, y, z = up) so that one of its axes points up
_TURN = {"+z": np.eye(3),
         "-z": np.diag([1.0, -1.0, -1.0]),                                 # 180 deg about x
         "+x": np.array([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]]),        # -90 deg about y: x -> z
         "-x": np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]]),        # +90 deg about y: -x -> z
         "+y": np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]]),        # +90 deg about x: y -> z
         "-y": np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])}        # -90 deg about x: -y -> z


@dataclass
class Rest:
    name: str            # "upright", "upside_down", "side_x+", ..., "lying"
    label: str           # for people: "on its 72 x 213 mm side"
    R: np.ndarray        # 3x3 base <- object at yaw 0 (rot_z(yaw) @ R for any heading)
    height_m: float      # center above the table
    footprint_m: tuple   # (along x, along y) at yaw 0
    period_deg: float | None   # the footprint repeats after this turn about z (None: round, any)


def rest_poses(obj_cfg: dict) -> dict[str, Rest]:
    """The object's stable resting faces. Box: one per face (6). Cylinder: upright, upside down, lying."""
    R_up = rotation_between(up_axis(obj_cfg), [0, 0, 1])                  # object frame -> upright frame
    dims = np.asarray(obj_cfg["dims_m"], float)                            # x, y, z of the UPRIGHT object
    sym = obj_cfg.get("grasp", {}).get("symmetric_yaw", False)
    out = {}
    if obj_cfg["shape"] == "cylinder":
        faces = {"upright": "+z", "upside_down": "-z", "lying": "+x"}
    else:
        faces = {"upright": "+z", "upside_down": "-z", "side_x+": "+x", "side_x-": "-x", "side_y+": "+y", "side_y-": "-y"}
    for name, ax in faces.items():
        R = _TURN[ax] @ R_up
        ext = np.abs(_TURN[ax]) @ dims                                     # extents along base x, y, z at yaw 0
        if obj_cfg["shape"] == "cylinder":
            period = None if (ax in ("+z", "-z") and sym) else 180.0
            label = {"upright": "standing", "upside_down": "upside down", "lying": "lying on its side"}[name]
        else:
            period = 90.0 if abs(ext[0] - ext[1]) < 1e-4 else 180.0
            down = sorted((ext[0] * 1000, ext[1] * 1000))
            label = {"upright": "standing", "upside_down": "upside down"}.get(
                name, f"on its {down[0]:.0f} x {down[1]:.0f} mm side ({ax[1]}{ax[0]} up)")
        out[name] = Rest(name, label, R, float(support_height(R, obj_cfg)), (float(ext[0]), float(ext[1])), period)
    return out


def current_rest(R_base_obj: np.ndarray, obj_cfg: dict, tol_deg: float = 25.0) -> str | None:
    """Which resting face an object in this orientation is on (None: none within tol_deg, e.g. leaning)."""
    best, err = None, tol_deg
    for name, r in rest_poses(obj_cfg).items():
        # compare the direction of "world up" in the object's own frame: yaw-free
        a, b = R_base_obj.T @ [0, 0, 1.0], r.R.T @ [0, 0, 1.0]
        if obj_cfg["shape"] == "cylinder" and name == "lying":                 # any axis direction in the table plane
            u = up_axis(obj_cfg)
            e = abs(90.0 - np.degrees(np.arccos(np.clip(abs(a @ u), 0.0, 1.0))))
        else:
            e = float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
        if e < err:
            best, err = name, e
    return best


# --- the hand: a few boxes in the TCP frame (config: gripper.hand_boxes) ---------------------------------------------
def hand_corners(gripper_cfg: dict, width_m: float) -> np.ndarray:
    """Corners [N, 3] of the hand's boxes in the TCP frame (fingers at the given opening)."""
    pts = []
    for b in gripper_cfg.get("hand_boxes", []):
        c, h = np.asarray(b["center"], float), np.asarray(b["half"], float).copy()
        if b.get("fingers"):                       # the finger box's half-width along y grows with the opening
            h[1] = width_m / 2 + h[1]
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    pts.append(c + h * [sx, sy, sz])
    return np.array(pts).reshape(-1, 3)


def place_poses(T_tcp_obj: np.ndarray, obj_cfg: dict, rest: Rest, goal_xy, table_z: float, clearance: float,
                yaws, gripper_cfg: dict, width_m: float, max_up: float = 0.2, base_xy=(0.0, 0.0)):
    """Hand poses that put the HELD object (T_tcp_obj) down on `rest` at goal_xy, one per object yaw in `yaws` that
    passes the checks -> list of (yaw, T_base_tcp, T_base_obj), hand pointing away from the robot first.
    Checks: the hand's approach axis (TCP z) not pointing up (z component <= max_up) and no hand corner below the
    table (+ 2 mm)."""
    R_to = T_tcp_obj[:3, :3]
    corners = hand_corners(gripper_cfg, width_m)
    out = []
    for yaw in yaws:
        R_obj = rot_z(yaw) @ rest.R
        R_tcp = R_obj @ R_to.T
        if R_tcp[2, 2] > max_up:
            continue
        p_obj = np.array([goal_xy[0], goal_xy[1], table_z + clearance + rest.height_m])
        p_tcp = p_obj - R_tcp @ T_tcp_obj[:3, 3]
        if len(corners) and float(np.min((corners @ R_tcp.T)[:, 2])) + p_tcp[2] < table_z + 0.002:
            continue
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_tcp, p_tcp
        To = np.eye(4)
        To[:3, :3], To[:3, 3] = R_obj, p_obj
        out.append((float(yaw), T, To))
    radial = np.asarray(goal_xy, float) - np.asarray(base_xy, float)
    radial = radial / (np.linalg.norm(radial) + 1e-9)

    def score(item):             # straight down best; a horizontal hand: pointing away from the robot (arm behind it)
        z = item[1][:3, 2]
        return -(-z[2] * 2.0 + float(z[:2] @ radial))
    return sorted(out, key=score)


def _resting_T(obj_cfg: dict, rest: Rest, yaw: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot_z(yaw) @ rest.R
    T[:3, 3] = [0.0, 0.0, rest.height_m]
    return T


@dataclass
class Move:
    frm: str
    to: str
    grasp_kind: str
    grasp_width_m: float
    hand: str            # "down" | "horizontal" | "tilted": how the hand is turned when it sets the object down


def transitions(obj_cfg: dict, gripper_cfg: dict, n_yaws: int = 8) -> list[Move]:
    """Every single pick-and-place move between resting faces this gripper can make (ignoring the arm's reach):
    a top-down grasp in face A whose turn to face B passes the checks of place_poses."""
    rests = rest_poses(obj_cfg)
    yaws = [2 * np.pi * k / n_yaws for k in range(n_yaws)]
    margin = float(gripper_cfg.get("open_margin_m", 0.0))
    moves = []
    for a in rests.values():
        T_obj = _resting_T(obj_cfg, a)
        grasps, _ = grasps_from_pose(T_obj, obj_cfg, [0.0, np.pi / 2], 0.0, 20.0, gripper_cfg["max_width_m"] - margin)
        for b in rests.values():
            if b.name == a.name:
                continue
            for g in grasps:
                ok = place_poses(g.T_tcp_obj, obj_cfg, b, (0.5, 0.0), 0.0, 0.005, yaws, gripper_cfg, g.width_m)
                if ok:
                    z = ok[0][1][2, 2]
                    hand = "down" if z < -0.9 else ("horizontal" if abs(z) < 0.2 else "tilted")
                    moves.append(Move(a.name, b.name, g.kind, g.width_m, hand))
                    break
    return moves


def plan_reorientation(obj_cfg: dict, gripper_cfg: dict, frm: str, to: str, moves=None) -> list[Move] | None:
    """Fewest moves from face `frm` to face `to` (breadth-first over transitions()); [] if already there, None if
    this gripper can't get there at all."""
    if frm == to:
        return []
    moves = transitions(obj_cfg, gripper_cfg) if moves is None else moves
    prev = {frm: None}
    q = deque([frm])
    while q:
        n = q.popleft()
        for m in moves:
            if m.frm == n and m.to not in prev:
                prev[m.to] = m
                if m.to == to:
                    path = []
                    while m is not None:
                        path.append(m)
                        m = prev[m.frm]
                    return path[::-1]
                q.append(m.to)
    return None

