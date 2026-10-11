"""The SO-101 profile: generated description, gripper model, numerical IK for top-down grasps, fixed-jaw shift.
Needs the fetched description (robots/so101/fetch_description.sh); skipped without it."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pnp_core import robot_description as rdesc
from pnp_core.catalog import load_robot, repo_root
from pnp_core.gripper import GripperModel
from pnp_core.object_grasp import ObjectGrasp, shift_for_fixed_jaw, top_down_R
from pnp_core.geometry import make_transform
from pnp_core.urdf_kin import Chain

CFG = load_robot("so101")
HAVE = (repo_root() / CFG["description"]["urdf"]).exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="SO-101 description not fetched")


@pytest.fixture(scope="module")
def chain():
    return Chain(rdesc.urdf(CFG, repo_root(), "mock"), CFG["frames"]["base"], CFG["frames"]["tcp"]).clamp_limits(
        CFG["arm"]["position_limits"], CFG["arm"]["limit_margin"])


def test_generated_urdf_has_tcp_meshes_and_ros2_control():
    root = ET.fromstring(rdesc.urdf(CFG, repo_root(), "isaac", 0.4))
    assert root.find("link[@name='so101_tcp']") is not None
    assert all(m.get("filename").startswith("file://") for m in root.iter("mesh"))
    rc = root.find("ros2_control")
    assert rc.find("hardware/plugin").text == "topic_based_ros2_control/TopicBasedSystem"
    assert [j.get("name") for j in rc.findall("joint")] == CFG["arm"]["joints"] + ["gripper"]
    mock = ET.fromstring(rdesc.urdf(CFG, repo_root(), "mock")).find("ros2_control/hardware/plugin").text
    assert mock == "mock_components/GenericSystem"


def test_generated_controller_configs_name_the_profile_joints():
    rc = rdesc.ros2_controllers(CFG)
    assert rc[CFG["arm"]["controller"]]["ros__parameters"]["joints"] == CFG["arm"]["joints"]
    assert rc[CFG["gripper"]["controller"]]["ros__parameters"]["joint"] == "gripper"
    mc = rdesc.moveit_controllers(CFG)["moveit_simple_controller_manager"]
    assert mc["controller_names"] == [CFG["arm"]["controller"], CFG["gripper"]["controller"]]


def test_jaw_gripper_width_roundtrip():
    g = GripperModel(CFG["gripper"])
    for w in (0.005, 0.016, 0.032, 0.05):
        assert abs(g.width(g.command(w)) - w) < 1e-4
    assert g.command(1.0) == pytest.approx(g.command(CFG["gripper"]["max_width_m"]))
    assert g.width(g.closed_command()) < 0.003
    fr3 = GripperModel(load_robot("fr3")["gripper"])
    assert fr3.command(0.06) == pytest.approx(0.03) and fr3.width(0.03) == pytest.approx(0.06)


def test_tcp_convention_matches_the_jaws(chain):
    """At a top-down pose the TCP z points down and its y runs along the jaws' closing direction."""
    q, pe, ae = chain.ik([0.22, 0.0, 0.03], target_R=top_down_R(0.0))
    assert pe < 1e-3 and ae < 0.5
    T = chain.fk(q)
    assert T[2, 2] < -0.999


@pytest.mark.parametrize("xy", [(0.18, 0.0), (0.25, 0.08), (0.22, -0.12), (0.28, 0.0)])
@pytest.mark.parametrize("yaw_deg", [0, 45, 90, -60])
def test_top_down_grasps_at_any_yaw_in_the_work_area(chain, xy, yaw_deg):
    """Any grasp yaw works, possibly turned by 180 deg (the same grasp of a box; pick & place tries both): wrist
    roll covers ~320 deg, so one of the two is always inside its range."""
    best = min((chain.ik([xy[0], xy[1], 0.012], target_R=top_down_R(np.radians(yaw_deg + flip)), seeds=10)
                for flip in (0, 180)), key=lambda r: r[1])
    q, pe, ae = best
    assert pe < 1e-3 and ae < 1.0, (pe, ae)
    assert np.all(q >= chain.lower - 1e-9) and np.all(q <= chain.upper + 1e-9)


def test_fixed_jaw_shift_puts_the_object_against_the_fixed_jaw():
    g = GripperModel(CFG["gripper"])
    T = make_transform(top_down_R(0.3), [0.2, 0.0, 0.01])
    grasp = ObjectGrasp("box", T, np.linalg.inv(T) @ T, 0.0078)        # object centered at the TCP
    shifted = shift_for_fixed_jaw([grasp], g.held_offset_y)[0]
    # held: the object's center is fixed_jaw_y + w/2 along TCP y (toward the moving jaw)
    assert shifted.T_tcp_obj[1, 3] == pytest.approx(CFG["gripper"]["fixed_jaw_y_m"] + 0.0078 / 2)
    assert np.allclose(shifted.T_base_tcp @ shifted.T_tcp_obj, T)          # the object itself didn't move
    fr3 = GripperModel(load_robot("fr3")["gripper"])
    assert shift_for_fixed_jaw([grasp], fr3.held_offset_y)[0] is grasp


def test_fixed_jaw_approach_clearance_keeps_the_held_offset():
    g = GripperModel(CFG["gripper"])
    T = make_transform(top_down_R(0.0), [0.2, 0.0, 0.01])
    grasp = ObjectGrasp("box", T, np.eye(4), 0.0158)
    s = shift_for_fixed_jaw([grasp], g.held_offset_y, g.approach_clearance)[0]
    off, clr = g.held_offset_y(0.0158), g.approach_clearance
    assert clr > 0
    assert s.T_tcp_obj[1, 3] == pytest.approx(off)                                  # where it sits once held
    assert (np.linalg.inv(T) @ s.T_base_tcp)[1, 3] == pytest.approx(-(off + clr))    # where the TCP comes down


def test_joint_space_board_views_for_the_overhead_camera(chain):
    """The SO-101 cell's camera calibration views: reachable by construction, board face-up and wholly in view."""
    from pnp_core.calibration import board_center_T, joint_space_board_views, mount_T
    from pnp_core.catalog import load_fixed_camera, load_yaml
    from pnp_core.geometry import look_at_optical
    targets = load_yaml("config/calibration_targets_so101.yaml")
    b, pz = targets["hand_board"], targets["hand_board"]["poses"]
    cam = load_fixed_camera(load_yaml("config/scenes/so101_lego.yaml"))["mount"]
    T_cam = make_transform(look_at_optical(cam["position"], cam["look_at"]), cam["position"])
    hand = Chain(rdesc.urdf(CFG, repo_root(), "mock"), CFG["frames"]["base"], b["mount"]["parent"])
    K = np.array([[615.0, 0, 320], [0, 615.0, 240], [0, 0, 1]])
    qs = joint_space_board_views(hand.fk, chain.lower, chain.upper, 12, np.random.default_rng(0), T_cam, K,
                                 (640, 480), mount_T(b), b, dist=tuple(pz["dist"]),
                                 max_view_angle_deg=pz["max_view_angle_deg"], ranges=pz["ranges"])
    assert len(qs) == 12
    for q in qs:
        Tb = hand.fk(q) @ mount_T(b) @ board_center_T(b)
        assert -Tb[2, 2] > np.cos(np.radians(60))            # printed face (-z) points up-ish, toward the camera
        assert np.all(q >= chain.lower - 1e-9) and np.all(q <= chain.upper + 1e-9)
    # rotations about different axes (hand-eye needs them): the hand orientations spread
    from pnp_core.calibration import rotation_spread_deg
    assert rotation_spread_deg([hand.fk(q) for q in qs]) > 30
