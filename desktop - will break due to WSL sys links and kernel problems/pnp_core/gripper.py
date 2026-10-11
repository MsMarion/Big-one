"""Gripper models: the pick logic speaks in jaw OPENING WIDTHS [m]; each gripper maps them to its joint.

    parallel  two mirrored fingers, one driven prismatic joint = the distance of ONE finger from the center
              (Franka Hand): command = width / 2, width = 2 * joint
    jaw       one moving jaw on a revolute joint against a fixed jaw (SO-101): width <-> angle through a table
              measured from the meshes (robot config gripper.width_table), interpolated both ways

The GripperCommand action takes the JOINT position, so robot_interface converts with these. Pure Python.
"""

from __future__ import annotations

import numpy as np


class GripperModel:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.type = cfg.get("type", "parallel")
        self.max_width = float(cfg["max_width_m"])
        self.joint = cfg["driven_joint"]
        if self.type == "jaw":
            t = cfg["width_table"]
            self.angles, self.widths = np.asarray(t["angle"], float), np.asarray(t["width"], float)
            if np.any(np.diff(self.widths) <= 0) or np.any(np.diff(self.angles) <= 0):
                raise ValueError("gripper.width_table must increase in both angle and width")
        elif self.type != "parallel":
            raise ValueError(f"unknown gripper type {self.type!r}")

    def command(self, width: float) -> float:
        """Joint position that opens the jaws to `width` (clamped to the gripper's range)."""
        w = float(np.clip(width, 0.0, self.max_width))
        if self.type == "parallel":
            return w / 2.0
        return float(np.interp(w, self.widths, self.angles, left=self.angles[0], right=self.angles[-1]))

    def closed_command(self) -> float:
        """Fully closed (what close_on_object commands: the object stops the jaws before)."""
        return 0.0 if self.type == "parallel" else float(self.angles[0])

    def width(self, joint_position: float) -> float:
        if self.type == "parallel":
            return 2.0 * float(joint_position)
        return float(np.interp(joint_position, self.angles, self.widths, left=self.widths[0], right=self.widths[-1]))

    @property
    def approach_clearance(self) -> float:
        """Fixed-jaw grippers: how far beside the object the fixed jaw comes down (then the moving jaw slides the
        object against it). 0 for mirrored fingers."""
        return float(self.cfg.get("fixed_jaw_clearance_m", 0.0)) if "fixed_jaw_y_m" in self.cfg else 0.0

    def held_offset_y(self, object_width: float) -> float:
        """Where a held object's center sits along the TCP's closing axis (y). 0 for mirrored fingers; with a
        fixed jaw the object is pushed against it: y = fixed_jaw_y + width / 2."""
        if "fixed_jaw_y_m" not in self.cfg:
            return 0.0
        return float(self.cfg["fixed_jaw_y_m"]) + object_width / 2.0
