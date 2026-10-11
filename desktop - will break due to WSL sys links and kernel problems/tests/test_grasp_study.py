"""Candidate grasps + gripper checks (pnp_core/grasp_study.py)."""
import numpy as np

from pnp_core.catalog import load_robot, load_yaml
from pnp_core.grasp_study import candidates, finger_length, static_rank
from pnp_core.reorient import hand_corners, rest_poses

CAT = load_yaml("config/objects.yaml")
FR3 = load_robot("fr3")["gripper"]
SO101 = load_robot("so101")["gripper"]


def test_finger_length_from_hand_boxes():
    assert 0.03 < finger_length(FR3) < 0.06
    assert 0.03 < finger_length(SO101) < 0.06


def test_box_candidates_respect_the_gripper():
    cs = candidates(CAT["cracker_box"], FR3, "upright")
    assert cs
    assert all(c.width_m <= FR3["max_width_m"] for c in cs)               # only the 72 mm side fits the 80 mm hand
    assert {round(c.width_m, 3) for c in cs} == {0.072}
    for c in cs:
        R = c.T_obj_tcp[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        r = rest_poses(CAT["cracker_box"])["upright"]
        T_obj = np.eye(4)
        T_obj[:3, :3], T_obj[:3, 3] = r.R, [0, 0, r.height_m]
        T = T_obj @ c.T_obj_tcp
        corners = hand_corners(FR3, c.width_m) @ T[:3, :3].T + T[:3, 3]
        assert corners[:, 2].min() >= 0.002 - 1e-9                       # the hand stays above the table


def test_straight_down_center_grasp_has_no_twist_and_ranks_first():
    best = static_rank(candidates(CAT["cracker_box"], FR3, "upright"))[0]
    assert best.tilt_deg == 0 and abs(best.offset_m) < 1e-9 and best.twist_nm < 1e-6
    off = [c for c in candidates(CAT["cracker_box"], FR3, "upright") if c.tilt_deg == 0 and abs(c.offset_m) > 0.02]
    assert off and all(c.twist_nm > 0.05 for c in off)                    # off-center: gravity twists it in the hand


def test_bricks_get_only_shallow_straight_down_grasps_on_the_so101():
    cs = candidates(CAT["lego_2x4"], SO101, "upright")
    assert cs and all(c.tilt_deg == 0 for c in cs)                        # a tilted jaw hits the desk
    assert all(c.depth_m < 0.012 for c in cs)
