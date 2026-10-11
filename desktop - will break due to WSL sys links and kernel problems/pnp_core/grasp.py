"""Top-down grasp, pre-grasp and place poses (plan section 2a step 5, 2b).

A top-down grasp has 4 degrees of freedom: x, y, z and yaw about the vertical axis. The gripper
TCP z-axis points straight down. For the TCP frame used here (fr3_hand_tcp / franka convention),
"pointing down" = rotate 180 deg about x, then apply the yaw about the world z-axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import matrix_to_quat, quat_to_matrix, rot_z, wrap_angle

R_DOWN = np.diag([1.0, -1.0, -1.0])     # rotation of 180 deg about x: tool z -> world -z


@dataclass
class Pose:
    position: np.ndarray    # (3,) in the base frame [m]
    quat_xyzw: np.ndarray   # (4,)

    @property
    def yaw(self) -> float:
        R = quat_to_matrix(self.quat_xyzw) @ R_DOWN.T
        return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class GraspPlan:
    pregrasp: Pose
    grasp: Pose
    lift: Pose
    open_width_m: float
    expected_width_m: float
    force_n: float


def top_down_quat(yaw: float) -> np.ndarray:
    return matrix_to_quat(rot_z(yaw) @ R_DOWN)


def choose_yaw(object_yaw: float, current_tcp_yaw: float, symmetric: bool) -> float:
    """Pick the grasp yaw that needs the least wrist rotation.
    Parallel grippers are 180-deg symmetric, so yaw and yaw+pi are the same grasp.
    Rotationally symmetric objects (cans) accept ANY yaw -> keep the current wrist yaw."""
    target = current_tcp_yaw if symmetric else object_yaw
    candidates = [wrap_angle(target), wrap_angle(target + np.pi)]
    return min(candidates, key=lambda y: abs(wrap_angle(y - current_tcp_yaw)))


def grasp_yaws(yaw: float, current_tcp_yaw: float, symmetric: bool) -> list:
    """Grasp yaws to try, best first. A parallel gripper is 180-deg symmetric; a rotationally symmetric object (the
    can) accepts ANY yaw, so also try turning the hand: with a box beside the object, the open hand may only fit one
    way. The pick (pick_place_node) and the workcell page's pickability check both use this list."""
    first = choose_yaw(yaw, current_tcp_yaw, symmetric)
    if not symmetric:
        return [first]
    return [choose_yaw(wrap_angle(first + d), current_tcp_yaw, False) for d in (0.0, np.pi / 2, np.pi / 4, -np.pi / 4)]


def plan_top_down_grasp(
    center_xy,
    z_top: float,
    yaw: float,
    grasp_cfg: dict,
    gripper_max_width: float,
    approach_height: float = 0.10,
    lift_height: float = 0.15,
) -> GraspPlan:
    """grasp_cfg = objects.yaml `grasp:` block (depth_below_top_m, width_m, force_n)."""
    q = top_down_quat(yaw)
    x, y = float(center_xy[0]), float(center_xy[1])
    z_grasp = z_top - float(grasp_cfg["depth_below_top_m"])
    return GraspPlan(
        pregrasp=Pose(np.array([x, y, z_grasp + approach_height]), q),
        grasp=Pose(np.array([x, y, z_grasp]), q),
        lift=Pose(np.array([x, y, z_grasp + lift_height]), q),
        open_width_m=float(gripper_max_width),
        expected_width_m=float(grasp_cfg["width_m"]),
        force_n=float(grasp_cfg["force_n"]),
    )


def place_poses(
    goal_xy,
    goal_yaw: float,
    table_z: float,
    object_height: float,
    grasp_depth_below_top: float,
    clearance: float = 0.005,
    approach_height: float = 0.10,
) -> tuple[Pose, Pose]:
    """(pre-place, place) TCP poses that put the object's bottom `clearance` above the table at goal."""
    q = top_down_quat(goal_yaw)
    z = table_z + object_height - grasp_depth_below_top + clearance
    x, y = float(goal_xy[0]), float(goal_xy[1])
    return Pose(np.array([x, y, z + approach_height]), q), Pose(np.array([x, y, z]), q)
