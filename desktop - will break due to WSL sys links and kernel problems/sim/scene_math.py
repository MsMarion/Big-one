"""Small orientation helpers shared by sim scripts. Only needs pxr (no running Isaac app), so it is unit-tested
(tests/test_upright_quat.py)."""

import numpy as np
from pxr import Gf


def upright_quat_wxyz(up_axis: str, yaw: float = 0.0) -> np.ndarray:
    """Orientation (w,x,y,z) that stands an asset upright, then yaws it about the world z-axis.

    Gotcha: Gf.Rotation composition `a * b` applies `a` FIRST, then `b` (row-vector convention).
    So "stand up, then yaw" is `q_up * q_yaw`. The reverse order tips the object over whenever yaw != 0.
    """
    up = {"Y": 90.0, "+Y": 90.0, "-Y": -90.0}           # rotation about x that brings the model's up to +z
    q_up = Gf.Rotation(Gf.Vec3d(1, 0, 0), up.get(up_axis.upper(), 0.0))   # "Z"/"+Z": already upright
    q = q_up * Gf.Rotation(Gf.Vec3d(0, 0, 1), float(np.degrees(yaw)))
    qq = q.GetQuat()
    return np.array([qq.GetReal(), *qq.GetImaginary()])
