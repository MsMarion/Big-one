import numpy as np
import pytest

from pnp_core.geometry import (
    R_LINK_OPTICAL,
    look_at_optical,
    optical_to_body,
    matrix_to_quat,
    optical_to_usd_camera,
    quat_to_matrix,
    rot_z,
    transform_points,
    make_transform,
    wrap_angle,
)


@pytest.mark.parametrize("yaw", [0.0, 0.3, -2.0, np.pi])
def test_quat_roundtrip(yaw):
    R = rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])
    assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-9)


def test_look_at_optical_axes():
    R = look_at_optical(eye=[0, 0, 1], target=[1, 0, 1])   # looking along +x, z up
    right, down, forward = R.T
    assert np.allclose(forward, [1, 0, 0])
    assert np.allclose(right, [0, -1, 0])     # right-hand side when facing +x is -y
    assert np.allclose(down, [0, 0, -1])
    assert np.isclose(np.linalg.det(R), 1.0)


def test_usd_camera_looks_down_minus_z():
    R_opt = look_at_optical([1, 1, 1], [0, 0, 0])
    R_usd = optical_to_usd_camera(R_opt)
    assert np.allclose(R_usd @ [0, 0, -1], R_opt @ [0, 0, 1])   # same viewing direction
    assert np.allclose(R_usd @ [0, 1, 0], R_opt @ [0, -1, 0])   # USD up = optical -y
    assert np.isclose(np.linalg.det(R_usd), 1.0)


def test_camera_link_axes():
    # camera at (1,0,1) looking along -x (back at the robot), level: camera_link x should point along -x, z up
    R_opt = look_at_optical([1, 0, 1], [0, 0, 1])
    R_link = optical_to_body(R_opt)
    assert np.allclose(R_link @ [1, 0, 0], [-1, 0, 0])   # body x = viewing direction
    assert np.allclose(R_link @ [0, 0, 1], [0, 0, 1])    # body z = up
    assert np.allclose(R_link @ R_LINK_OPTICAL, R_opt)   # chaining link->optical gives back optical
    # rpy(-pi/2, 0, -pi/2) is the standard realsense link->optical rotation
    cr, sr, cy, sy = np.cos(-np.pi / 2), np.sin(-np.pi / 2), np.cos(-np.pi / 2), np.sin(-np.pi / 2)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    assert np.allclose(Rz @ Rx, R_LINK_OPTICAL)


def test_transform_points():
    T = make_transform(rot_z(np.pi / 2), [1, 2, 3])
    assert np.allclose(transform_points(T, np.array([[1.0, 0, 0]])), [[1, 3, 3]])


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-np.pi), np.pi)
