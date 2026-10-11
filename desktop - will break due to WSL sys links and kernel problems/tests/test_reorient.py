"""Resting faces, turned placements and the reorientation graph (pnp_core/reorient.py)."""
import numpy as np
import pytest

from pnp_core.catalog import load_robot, load_yaml
from pnp_core.object_grasp import grasps_from_pose, support_height, up_axis
from pnp_core.reorient import current_rest, place_poses, plan_reorientation, rest_poses, transitions

CAT = load_yaml("config/objects.yaml")
FR3 = load_robot("fr3")["gripper"]
SO101 = load_robot("so101")["gripper"]


@pytest.mark.parametrize("name", ["tomato_soup_can", "cracker_box", "lego_2x4", "lego_1x2"])
def test_rest_poses_are_rotations_resting_on_the_table(name):
    obj = CAT[name]
    for r in rest_poses(obj).values():
        assert np.allclose(r.R @ r.R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(r.R), 1.0)
        assert np.isclose(r.height_m, support_height(r.R, obj))
        assert current_rest(r.R, obj) == r.name or (obj["shape"] == "cylinder" and r.name == "lying")


def test_box_faces():
    rests = rest_poses(CAT["cracker_box"])                  # 164 x 72 x 213 mm upright
    assert set(rests) == {"upright", "upside_down", "side_x+", "side_x-", "side_y+", "side_y-"}
    assert np.isclose(rests["upright"].height_m, 0.2134 / 2, atol=1e-3)
    assert np.isclose(rests["side_x+"].height_m, 0.164 / 2, atol=1e-3)      # on its narrow 72 x 213 side
    assert np.isclose(rests["side_y+"].height_m, 0.0718 / 2, atol=1e-3)     # flat on its big face
    up = rests["upside_down"].R @ up_axis(CAT["cracker_box"])
    assert np.allclose(up, [0, 0, -1], atol=1e-9)


def test_fr3_box_graph():
    """The Franka hand (80 mm) holds the box only across its 72 mm side: it can stand it on the narrow side and,
    via that, upside down -- never flat (that face is the one the fingers hold: a finger would be under the box)."""
    obj = CAT["cracker_box"]
    moves = transitions(obj, FR3)
    assert {(m.frm, m.to) for m in moves if m.frm == "upright"} == {("upright", "side_x+"), ("upright", "side_x-")}
    assert all(m.hand == "horizontal" for m in moves)
    path = plan_reorientation(obj, FR3, "upright", "upside_down", moves)
    assert [m.to for m in path] in (["side_x+", "upside_down"], ["side_x-", "upside_down"])
    assert plan_reorientation(obj, FR3, "upright", "side_y+", moves) is None
    assert plan_reorientation(obj, FR3, "upright", "upright", moves) == []


def test_fr3_can_graph():
    obj = CAT["tomato_soup_can"]
    path = plan_reorientation(obj, FR3, "upright", "upside_down")
    assert [m.to for m in path] == ["lying", "upside_down"]


def test_so101_brick_center_grasp_cannot_flip():
    """The SO-101's jaws are 48 mm wide: a brick held at its center is too low to be stood on its end without a
    jaw edge hitting the desk (the measured hand boxes, tools/hand_geometry.py)."""
    assert transitions(CAT["lego_2x4"], SO101) == []


def test_place_pose_keeps_the_object_where_asked():
    obj = CAT["cracker_box"]
    up = rest_poses(obj)
    T0 = np.eye(4)
    T0[:3, :3], T0[:3, 3] = up["upright"].R, [0.5, 0.2, up["upright"].height_m]
    g = grasps_from_pose(T0, obj, [0.0], 0.0, 20.0, 0.08)[0][0]
    out = place_poses(g.T_tcp_obj, obj, up["side_x+"], (0.45, -0.2), 0.0, 0.005,
                      [k * np.pi / 4 for k in range(8)], FR3, g.width_m)
    assert out
    for yaw, T_tcp, T_obj in out:
        assert np.allclose(T_tcp @ g.T_tcp_obj, T_obj, atol=1e-9)              # the hand holds it like at the pick
        assert np.allclose(T_obj[:3, 3], [0.45, -0.2, 0.005 + 0.082], atol=1e-3)
        assert abs(T_tcp[2, 2]) < 0.2                                         # the hand is horizontal
        assert current_rest(T_obj[:3, :3], obj) == "side_x+"
    # upright -> upside down in ONE move: the hand would point up: no pose
    assert place_poses(g.T_tcp_obj, obj, up["upside_down"], (0.45, -0.2), 0.0, 0.005, [0.0, np.pi / 2], FR3,
                       g.width_m) == []
