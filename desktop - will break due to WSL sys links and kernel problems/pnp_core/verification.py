"""Did it work? Checks that use only signals available on BOTH the sim and the real robot
(finger joint positions from /joint_states, re-detection by the camera). Plan section 2c."""

from __future__ import annotations

from enum import Enum

import numpy as np


class GraspState(str, Enum):
    HOLDING = "holding"        # fingers stopped at ~object width
    EMPTY = "empty"            # fingers closed fully: missed the object
    UNEXPECTED = "unexpected"  # stopped at a width that doesn't match the object (wrong object / bad grasp)


def finger_width(finger_positions) -> float:
    """Franka-style: each finger joint is the distance from the center, width = sum."""
    return float(np.sum(finger_positions))


def classify_grasp(width: float, expected_width: float, tol: float = 0.01, empty_below: float = 0.005) -> GraspState:
    if width < empty_below:
        return GraspState.EMPTY
    if abs(width - expected_width) <= tol:
        return GraspState.HOLDING
    return GraspState.UNEXPECTED


def dropped(width_history, expected_width: float, tol: float = 0.01) -> bool:
    """True if the gripper was holding and then closed further (object slipped out)."""
    w = np.asarray(width_history, dtype=float)
    return bool(w.size and w.min() < expected_width - tol)


def placement_error(actual_xy, goal_xy) -> float:
    return float(np.linalg.norm(np.asarray(actual_xy, float) - np.asarray(goal_xy, float)))
