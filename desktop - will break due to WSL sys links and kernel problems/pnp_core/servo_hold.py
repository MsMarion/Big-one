"""Make a loaded hobby servo actually REACH its goal: an outer integral loop around the servo's own P controller.

An STS3215 is a position servo with (mostly) P control: under load it stops short of the goal, by however far its
gain needs to make the holding torque. On the real SO-101 (12 V 2 A, LeRobot's gain 16 -> our 32) that was 1-2 deg at
the shoulder and elbow: ~1 cm at the gripper, and every MoveIt move ended "CONTROL_FAILED" against a 1 deg tolerance.
The ENCODER sees it (unlike link flex), so a slow outer loop can remove it: while a joint holds still, add the
remaining error to what is sent to the servo, until the encoder reads the goal.

Safety, by construction:
- only while the joint is SETTLED: its goal not moving and its measured speed small -- never fights a motion
- not when the error is large (> blocked): that's a joint pushing against something, not sag -- don't push harder
- the correction is clamped (max), applied to the ARM joints only (never the gripper: its force stays as set)
- reset at every torque-on
"""
from __future__ import annotations

import numpy as np


class HoldIntegrator:
    def __init__(self, names, ki: float = 2.0, max_rad: float = 0.09, settle_vel: float = 0.05,
                 blocked_rad: float = 0.14):
        self.names = list(names)
        self.ki, self.max, self.settle_vel, self.blocked = ki, max_rad, settle_vel, blocked_rad
        self.off = {n: 0.0 for n in self.names}

    def reset(self):
        self.off = {n: 0.0 for n in self.names}

    def update(self, goal: dict, q: dict, dq: dict, moving: dict, dt: float) -> dict:
        """goal: where each joint should be (rad), q / dq: measured, moving: is its goal still changing.
        -> the offset to ADD to each joint's goal before it is sent to the servo."""
        if self.ki <= 0:
            return self.off
        for n in self.names:
            if n not in goal or n not in q:
                continue
            err = goal[n] - q[n]
            if moving.get(n, False) or abs(dq.get(n, 0.0)) > self.settle_vel or abs(err) > self.blocked:
                continue
            self.off[n] = float(np.clip(self.off[n] + self.ki * err * dt, -self.max, self.max))
        return self.off
