import numpy as np

from pnp_core.catalog import load_object, load_robot
from pnp_core.geometry import quat_to_matrix
from pnp_core.grasp import choose_yaw, place_poses, plan_top_down_grasp, top_down_quat
from pnp_core.verification import GraspState, classify_grasp, dropped, finger_width, placement_error


def test_top_down_quat_points_tool_z_down():
    for yaw in (0.0, 0.7, -2.5):
        R = quat_to_matrix(top_down_quat(yaw))
        assert np.allclose(R @ [0, 0, 1], [0, 0, -1])
        assert np.isclose(np.arctan2(R[1, 0], R[0, 0]), yaw)


def test_choose_yaw():
    assert np.isclose(choose_yaw(1.0, current_tcp_yaw=0.2, symmetric=True), 0.2)
    # a box at yaw 3.0 is the same grasp as 3.0 - pi = -0.14, which is closer to 0
    assert np.isclose(choose_yaw(3.0, current_tcp_yaw=0.0, symmetric=False), 3.0 - np.pi)


def test_plan_for_soup_can_from_catalog():
    can = load_object("tomato_soup_can")
    robot = load_robot("fr3")
    plan = plan_top_down_grasp([0.5, 0.2], z_top=0.101, yaw=0.0, grasp_cfg=can["grasp"],
                               gripper_max_width=robot["gripper"]["max_width_m"])
    assert np.isclose(plan.grasp.position[2], 0.101 - 0.03)
    assert np.isclose(plan.pregrasp.position[2] - plan.grasp.position[2], 0.10)
    assert plan.expected_width_m < plan.open_width_m
    assert np.isclose(plan.grasp.yaw, 0.0)


def test_place_pose_height():
    pre, place = place_poses([0.5, -0.3], 0.0, table_z=0.0, object_height=0.101,
                             grasp_depth_below_top=0.03, clearance=0.005)
    assert np.isclose(place.position[2], 0.101 - 0.03 + 0.005)
    assert pre.position[2] > place.position[2]


def test_grasp_classification():
    assert finger_width([0.033, 0.033]) == 0.066
    assert classify_grasp(0.066, 0.066) is GraspState.HOLDING
    assert classify_grasp(0.001, 0.066) is GraspState.EMPTY
    assert classify_grasp(0.030, 0.066) is GraspState.UNEXPECTED
    assert dropped([0.066, 0.065, 0.040], 0.066)
    assert not dropped([0.066, 0.065, 0.066], 0.066)
    assert np.isclose(placement_error([0.51, -0.3], [0.5, -0.3]), 0.01)
