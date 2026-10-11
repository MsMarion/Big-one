import numpy as np

from pnp_core.catalog import load_object
from pnp_core.geometry import make_transform, rot_z
from pnp_core.grasp import place_poses, plan_top_down_grasp
from pnp_core.object_grasp import (grasps_from_pose, nominal_grasp_offset, object_extents, object_yaw,
                                   rotation_error_deg,
                                   place_tcp_poses,
                                   place_tcp_yaws, support_height,
                                   tcp_yaw_of, tilt_deg, upright_pose)

CAN = load_object("tomato_soup_can")
H, R_CAN, D = CAN["dims_m"][2], CAN["dims_m"][0] / 2, CAN["grasp"]["depth_below_top_m"]


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def test_upright_pose_matches_the_sim_convention():
    # YCB assets are Y-up; the sim stands them up with +90 deg about x (mesh y -> world z)
    T = upright_pose(CAN, [0.5, 0.1], H / 2)
    assert np.allclose(T[:3, :3], rot_x(np.pi / 2))
    assert np.isclose(tilt_deg(T[:3, :3], CAN), 0.0)
    assert np.isclose(support_height(T[:3, :3], CAN), H / 2)


def test_upright_can_gives_the_same_grasp_as_the_detector_path():
    T = upright_pose(CAN, [0.5, 0.1], H / 2, yaw=1.0)            # the can's own yaw doesn't matter
    gs, why = grasps_from_pose(T, CAN, [0.3, 0.3 + np.pi / 2])
    assert why.startswith("cap") and [g.kind for g in gs] == ["cap", "cap"]
    old = plan_top_down_grasp([0.5, 0.1], H, 0.3, CAN["grasp"], 0.08)
    new = gs[0].plan(0.08, CAN["grasp"]["force_n"])
    for a, b in ((old.grasp, new.grasp), (old.pregrasp, new.pregrasp), (old.lift, new.lift)):
        assert np.allclose(a.position, b.position) and np.allclose(abs(a.quat_xyzw @ b.quat_xyzw), 1.0)
    assert np.isclose(tcp_yaw_of(gs[1].T_base_tcp), 0.3 + np.pi / 2)
    assert np.isclose(new.expected_width_m, 2 * R_CAN)


def test_nominal_place_matches_place_poses():
    out = place_tcp_poses(nominal_grasp_offset(CAN), CAN, [0.4, -0.3], 0.0, 0.005, [0.2])
    yaw, pre, place, T_obj = out[0]
    _, old = place_poses([0.4, -0.3], 0.2, 0.0, H, D, 0.005)
    assert np.allclose(place.position, old.position)
    assert np.isclose(T_obj[2, 3] - H / 2, 0.005)                  # the can's bottom 5 mm above the table
    assert np.isclose(pre.position[2] - place.position[2], 0.10)


def test_place_uses_the_measured_grasp_offset():
    # the can ended up 5 mm deeper in the hand than planned -> release 5 mm lower, bottom still at the clearance
    T = upright_pose(CAN, [0.5, 0.0], H / 2)
    g = grasps_from_pose(T, CAN, [0.0])[0][0]
    deeper = make_transform(np.eye(3), [0, 0, 0.005]) @ g.T_tcp_obj   # object 5 mm further along tcp z (down)
    _, _, p_nom, _ = place_tcp_poses(g.T_tcp_obj, CAN, [0.4, -0.3], 0.0, 0.005, [0.0])[0]
    _, _, p_deep, T_obj = place_tcp_poses(deeper, CAN, [0.4, -0.3], 0.0, 0.005, [0.0])[0]
    assert np.isclose(p_deep.position[2] - p_nom.position[2], 0.005)
    assert np.isclose(T_obj[2, 3] - H / 2, 0.005)


def test_held_pose_round_trip():
    # grasp -> carry -> place: T_base_tcp_place @ T_tcp_obj is exactly the object pose place_tcp_poses promised
    T = upright_pose(CAN, [0.55, 0.12], H / 2, yaw=0.4)
    g = grasps_from_pose(T, CAN, [0.7])[0][0]
    assert np.allclose(g.T_base_tcp @ g.T_tcp_obj, T)
    for yaw, _, place, T_obj in place_tcp_poses(g.T_tcp_obj, CAN, [0.4, -0.3], 0.0, 0.005, [0.0, 1.2]):
        T_tcp = make_transform(rot_z(yaw) @ np.diag([1.0, -1, -1]), place.position)
        assert np.allclose(T_tcp @ g.T_tcp_obj, T_obj)
        assert np.allclose(T_obj[:2, 3], [0.4, -0.3])


def test_small_estimated_tilt_still_comes_straight_down():
    T = upright_pose(CAN, [0.5, 0.0], H / 2)
    T[:3, :3] = rot_x(np.radians(3)) @ T[:3, :3]                 # pose estimate 3 deg off vertical
    gs, why = grasps_from_pose(T, CAN, [0.0])
    assert gs[0].kind == "cap"
    assert np.allclose(gs[0].T_base_tcp[:3, 2], [0, 0, -1])       # approach exactly vertical


def test_upside_down_can_is_grasped_at_the_end_that_is_up():
    T = make_transform(rot_x(-np.pi / 2), [0.5, 0.0, H / 2])        # mesh y -> world -z
    assert np.isclose(tilt_deg(T[:3, :3], CAN), 180.0)
    gs, _ = grasps_from_pose(T, CAN, [0.0])
    assert gs[0].kind == "cap" and np.isclose(gs[0].T_base_tcp[2, 3], H - D)


def test_lying_can_gets_a_side_grasp_across_the_diameter():
    R = rot_z(0.5) @ rot_x(0.0)                                   # mesh y (the axis) horizontal, along yaw 0.5+90deg
    T = make_transform(R, [0.5, 0.1, R_CAN])
    assert np.isclose(tilt_deg(R, CAN), 90.0)
    assert np.isclose(support_height(R, CAN), R_CAN)
    gs, why = grasps_from_pose(T, CAN, [0.0], current_yaw=0.0)
    assert why.startswith("side") and len(gs) == 1
    axis = R @ [0, 1, 0]
    assert np.isclose(abs(gs[0].T_base_tcp[:3, 0] @ axis), 1.0)    # palm (tcp x) along the axis
    assert np.isclose(gs[0].T_base_tcp[:3, 1] @ axis, 0.0)         # fingers close across it
    assert np.allclose(gs[0].T_base_tcp[:3, 3], [0.5, 0.1, R_CAN])


def test_halfway_tilted_can_is_refused_with_a_reason():
    T = make_transform(rot_x(np.radians(45)) @ rot_x(np.pi / 2), [0.5, 0.0, 0.06])
    gs, why = grasps_from_pose(T, CAN, [0.0])
    assert gs == [] and "tilted" in why


def test_non_symmetric_object_is_turned_to_the_goal_yaw():
    box_like = dict(CAN, grasp=dict(CAN["grasp"], symmetric_yaw=False))
    T = upright_pose(box_like, [0.5, 0.0], H / 2, yaw=0.3)
    g = grasps_from_pose(T, box_like, [1.0])[0][0]                 # gripper at yaw 1.0, object at 0.3
    assert np.isclose(object_yaw(T[:3, :3], box_like), 0.3)
    (yaw,) = place_tcp_yaws(g.T_tcp_obj, box_like, -0.5, [9.9])
    _, _, _, T_obj = place_tcp_poses(g.T_tcp_obj, box_like, [0.4, -0.3], 0.0, 0.005, [yaw])[0]
    assert np.isclose(object_yaw(T_obj[:3, :3], box_like), -0.5)
    assert place_tcp_yaws(g.T_tcp_obj, CAN, -0.5, [0.1, 0.2]) == [0.1, 0.2]    # symmetric: any yaw


# --- boxes (the cracker box: 164 x 213.4 x 71.8 mm along mesh x/y/z, Y-up) ------------------------------------
BOX = load_object("cracker_box")
BW, BT, BH = BOX["dims_m"]                         # upright width (x), thickness (y), height (z)


def test_box_extents_in_its_own_frame():
    assert np.allclose(object_extents(BOX), [BW, BH, BT])


def test_standing_box_is_grasped_across_its_thickness():
    for yaw in (0.0, 0.6, -2.0):
        T = upright_pose(BOX, [0.5, 0.1], BH / 2, yaw=yaw)
        gs, why = grasps_from_pose(T, BOX, [0.0], current_yaw=0.0)
        assert why.startswith("box") and len(gs) == 1                 # the 164 mm side never fits the 80 mm hand
        g = gs[0]
        thin = T[:3, :3] @ [0, 0, 1.0]                                 # mesh z = the 71.8 mm side
        assert np.isclose(abs(g.T_base_tcp[:3, 1] @ thin), 1.0)        # fingers close across it
        assert np.allclose(g.T_base_tcp[:3, 2], [0, 0, -1])            # straight down
        assert np.isclose(g.T_base_tcp[2, 3], BH - BOX["grasp"]["depth_below_top_m"])
        assert np.isclose(g.width_m, BT)
        assert np.allclose(g.T_base_tcp @ g.T_tcp_obj, T)


def test_box_grasp_yaw_follows_the_box_and_prefers_the_smaller_wrist_turn():
    T = upright_pose(BOX, [0.5, 0.1], BH / 2, yaw=np.pi / 2)
    a = grasps_from_pose(T, BOX, [0.0], current_yaw=1.4)[0][0]
    b = grasps_from_pose(T, BOX, [0.0], current_yaw=-1.4)[0][0]
    assert np.isclose(tcp_yaw_of(a.T_base_tcp), np.pi / 2) and np.isclose(tcp_yaw_of(b.T_base_tcp), -np.pi / 2)


def test_box_on_its_narrow_side_and_flat():
    on_side = make_transform(rot_z(0.3) @ rot_y(np.pi / 2) @ rot_x(np.pi / 2), [0.5, 0, BW / 2])   # mesh x vertical
    gs, why = grasps_from_pose(on_side, BOX, [0.0])
    assert why.startswith("box") and np.isclose(gs[0].width_m, BT)
    assert np.isclose(gs[0].T_base_tcp[2, 3], BW - BOX["grasp"]["depth_below_top_m"])
    flat = make_transform(np.eye(3), [0.5, 0, BT / 2])                     # mesh z vertical: lying on the print
    gs, why = grasps_from_pose(flat, BOX, [0.0])
    assert gs == [] and "no side fits" in why
    tipped = make_transform(rot_x(np.radians(35)) @ rot_x(np.pi / 2), [0.5, 0, 0.1])
    assert grasps_from_pose(tipped, BOX, [0.0])[0] == []


def test_box_too_thick_for_the_margin_is_refused():
    thick = dict(BOX, dims_m=[0.164, 0.078, 0.2134])                      # 78 mm + 2 x 2 mm > 80 mm
    gs, why = grasps_from_pose(upright_pose(thick, [0.5, 0], 0.1), thick, [0.0])
    assert gs == [] and "no side fits" in why


def test_box_is_placed_at_the_goal_yaw_either_way_round():
    T = upright_pose(BOX, [0.5, 0.1], BH / 2, yaw=0.4)
    g = grasps_from_pose(T, BOX, [0.0])[0][0]
    yaws = place_tcp_yaws(g.T_tcp_obj, BOX, 0.0, [9.9], current_yaw=0.0)
    assert len(yaws) == 2 and abs(yaws[0]) <= abs(yaws[1])                 # 180 deg period: two ways, small turn first
    for y in yaws:
        _, _, place, T_obj = place_tcp_poses(g.T_tcp_obj, BOX, [0.4, -0.3], 0.0, 0.005, [y])[0]
        assert np.isclose(np.sin(object_yaw(T_obj[:3, :3], BOX)), 0.0, atol=1e-9)   # yaw 0 or 180 deg
        assert np.isclose(T_obj[2, 3], BH / 2 + 0.005)


def test_rotation_error_respects_symmetry():
    can_up = upright_pose(CAN, [0.5, 0.0], H / 2)[:3, :3]
    spun = rot_z(1.0) @ can_up                                                  # a can spun about its axis: same pose
    assert np.isclose(rotation_error_deg(can_up, spun, CAN), 0.0, atol=1e-6)
    tilted = rot_x(np.radians(5)) @ can_up
    assert np.isclose(rotation_error_deg(can_up, tilted, CAN), 5.0, atol=1e-6)
    box = upright_pose(BOX, [0.5, 0.0], BH / 2)[:3, :3]
    assert np.isclose(rotation_error_deg(box, rot_z(np.pi) @ box, BOX), 0.0, atol=1e-6)   # 180 deg: same box
    assert np.isclose(rotation_error_deg(box, rot_z(np.radians(3)) @ box, BOX), 3.0, atol=1e-6)
