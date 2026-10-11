"""pnp_core/grasp_scoring.py + object_grasp.with_flips + the trial log (pnp_core/trials.py)."""
import math

import numpy as np
import pytest

from pnp_core import trials
from pnp_core.catalog import load_yaml
from pnp_core.grasp_scoring import closing_side, joint_margin, score, tried_match, yaw_tolerance_deg
from pnp_core.object_grasp import grasps_from_pose, tcp_yaw_of, upright_pose, with_flips


def test_yaw_tolerance_shrinks_with_length():
    # a 2x4 (long along the jaws) tolerates less yaw error than a 2x2 with the same room beside it
    assert yaw_tolerance_deg(0.0158, 0.0318, 0.003) < yaw_tolerance_deg(0.0158, 0.0158, 0.003)
    assert 8.0 < yaw_tolerance_deg(0.0158, 0.0318, 0.003) < 14.0
    assert yaw_tolerance_deg(0.0158, 0.0318, 0.0) == 0.0


def test_score_prefers_clearance_margin_and_untried():
    a = {"clearance": 0.03, "joint_margin": 0.5, "yaw_tol": 12.0}
    b = {"clearance": 0.005, "joint_margin": 0.1, "yaw_tol": 12.0}
    assert score(a) > score(b)
    assert score({**a, "tried": True}) < score(b)


def test_score_narrow_side_beats_a_little_more_room():
    # the demo's 2x4: across 15.8 mm with 25 mm clearance vs across its 31.8 mm length with 31 mm
    narrow = {"narrow": 1.0, "clearance": 0.025, "joint_margin": 0.6, "yaw_tol": 11.0}
    wide = {"narrow": 15.8 / 31.8, "clearance": 0.031, "joint_margin": 0.6, "yaw_tol": 45.0}
    assert score(narrow) > score(wide)


def test_joint_margin():
    assert joint_margin([0.0, 1.0], ["a", "b"], {"a": [-1, 1], "b": [-2, 1.2]}) == pytest.approx(0.2)


def test_flips_turn_the_hand_keep_the_object():
    obj = load_yaml("config/objects.yaml")["lego_2x4"]
    T = upright_pose(obj, (0.2, 0.0), 0.0057, 0.3)
    gs, _ = grasps_from_pose(T, obj, [0.0], 0.0, 20.0, 0.055)
    fl = with_flips(gs)
    assert len(fl) == 2 * len(gs)
    a, b = fl[0], fl[len(gs)]
    assert abs(math.remainder(tcp_yaw_of(a.T_base_tcp) - tcp_yaw_of(b.T_base_tcp), 2 * math.pi)) == pytest.approx(math.pi)
    assert np.allclose(a.T_base_tcp @ a.T_tcp_obj, b.T_base_tcp @ b.T_tcp_obj)       # the same object pose
    assert closing_side(a.T_base_tcp, T)[0] == closing_side(b.T_base_tcp, T)[0]      # same axis ...
    assert closing_side(a.T_base_tcp, T) != closing_side(b.T_base_tcp, T)            # ... the other jaw on it
    assert tried_match(tcp_yaw_of(a.T_base_tcp), [tcp_yaw_of(a.T_base_tcp) + 0.1])
    assert not tried_match(tcp_yaw_of(a.T_base_tcp), [tcp_yaw_of(b.T_base_tcp)])


def test_trial_log_roundtrip(tmp_path):
    c = trials.connect(str(tmp_path / "t.sqlite"))
    run = trials.start_run(c, "s", 1, "quick", {"grasp_scoring": True})
    assert trials.current_run(c) == run
    a = trials.add_attempt(c, run, object="lego_2x2", result="SUCCESS", failure="ok", corrections=[[1.0, 3.0]])
    trials.update_attempt(c, a, true_x=0.2)
    trials.add_candidates(c, run, a, [{"idx": 0, "score": 1.5, "chosen": 1}])
    assert trials.last_attempt(c, run)["true_x"] == 0.2
    assert trials.classify_failure("GRASP_FAILED", "grasp unexpected (width 23 mm)") == "unexpected_width"
    assert trials.classify_failure("GRASP_FAILED", "approach blocked 5 mm above") == "bumped"
    trials.close_run(c, run)
    assert trials.current_run(c) != run                       # closed: a new 'manual' run
