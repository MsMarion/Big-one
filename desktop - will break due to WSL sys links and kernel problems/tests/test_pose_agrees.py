"""pnp_core.object_grasp.pose_agrees: when is a 6-D pose trusted over the detector (grasp_source auto)?"""
import numpy as np

from pnp_core.catalog import load_object
from pnp_core.object_grasp import pose_agrees, upright_pose

BRICK, CAN = load_object("lego_2x4"), load_object("tomato_soup_can")


def test_the_limit_scales_with_the_object():
    can = upright_pose(CAN, [0.5, 0.0], 0.05, 0.0)
    assert pose_agrees(can, [0.5 + 0.014, 0.0], 0.0, CAN)[0]                    # FR3 can: 15 mm as before
    brick = upright_pose(BRICK, [0.19, 0.06], 0.003, 0.0)
    ok, why = pose_agrees(brick, [0.19 + 0.0134, 0.06], 0.0, BRICK)             # the stale pose seen in the sim
    assert not ok and "limit 6 mm" in why
    assert pose_agrees(brick, [0.19 + 0.004, 0.06], 0.0, BRICK)[0]


def test_brick_yaw_must_agree_modulo_180():
    brick = upright_pose(BRICK, [0.19, 0.06], 0.003, 0.3)
    assert not pose_agrees(brick, [0.19, 0.06], 0.3 + np.radians(66), BRICK)[0]
    assert pose_agrees(brick, [0.19, 0.06], 0.3 + np.pi - np.radians(5), BRICK)[0]   # the same outline, turned 180
    assert pose_agrees(brick, [0.19, 0.06], None, BRICK)[0]                          # no detector yaw: xy only


def test_symmetric_objects_skip_the_yaw_check():
    can = upright_pose(CAN, [0.5, 0.0], 0.05, 0.0)
    assert pose_agrees(can, [0.5, 0.0], 1.0, CAN)[0]
