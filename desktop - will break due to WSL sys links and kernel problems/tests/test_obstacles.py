import numpy as np
import pytest

from pnp_core.catalog import load_yaml
from pnp_core.obstacles import footprint_distance, layout_problems, sample_layout, segment_hits_box

BOX = {"id": "b", "size": [0.2, 0.1, 0.3], "center": [1.0, 0.0, 0.15], "yaw": 0.0}


def test_footprint_distance_axis_aligned():
    assert footprint_distance((1.0, 0.0), BOX) == 0.0                     # inside
    assert footprint_distance((1.3, 0.0), BOX) == pytest.approx(0.2)      # 0.3 - half length 0.1
    assert footprint_distance((1.0, 0.25), BOX) == pytest.approx(0.2)     # 0.25 - half width 0.05


def test_footprint_distance_rotated():
    rot = dict(BOX, yaw=np.pi / 2)                                         # long side now along y
    assert footprint_distance((1.0, 0.25), rot) == pytest.approx(0.15)
    assert footprint_distance((1.3, 0.0), rot) == pytest.approx(0.25)


def test_segment_hits_box():
    assert segment_hits_box(np.array([0.0, 0.0, 0.1]), np.array([2.0, 0.0, 0.1]), BOX)       # straight through
    assert not segment_hits_box(np.array([0.0, 0.0, 0.5]), np.array([2.0, 0.0, 0.5]), BOX)   # above it
    assert not segment_hits_box(np.array([0.0, 0.0, 0.1]), np.array([0.85, 0.0, 0.1]), BOX)  # stops before
    assert segment_hits_box(np.array([0.0, 0.0, 0.1]), np.array([0.85, 0.0, 0.1]), BOX, pad=0.06)


@pytest.fixture(scope="module")
def scene():
    s = load_yaml("config/scenes/basic.yaml")
    cam = load_yaml("config/camera_d435i.yaml")["mount"]["position"]
    top = s["table"]["center"][2] + s["table"]["size"][2] / 2
    return s, cam, top


def test_random_layouts_are_solvable(scene):
    """Every sampled layout satisfies all the 'solvable' rules, and walls do land between can and goal."""
    s, cam, top = scene
    rng = np.random.default_rng(0)
    n_boxes, n_between = 0, 0
    for _ in range(200):
        can = [rng.uniform(*s["target"]["spawn_region"]["x"]), rng.uniform(*s["target"]["spawn_region"]["y"])]
        goal = [rng.uniform(*s["goal"]["region"]["x"]), rng.uniform(*s["goal"]["region"]["y"])]
        if np.hypot(can[0] - goal[0], can[1] - goal[1]) < s["goal"]["min_separation_m"]:
            continue
        boxes = sample_layout(rng, s["obstacles"], can, goal, cam, s["table"], top, 0.1018)
        assert layout_problems(boxes, can, goal, cam, s["table"], top, 0.1018, s["obstacles"]) == []
        assert len({b["id"] for b in boxes}) == len(boxes)
        n_boxes += len(boxes)
        for b in boxes:
            assert b["center"][2] == pytest.approx(top + b["size"][2] / 2)
            if b["id"] == "wall":
                mid = (np.asarray(can) + np.asarray(goal)) / 2
                n_between += np.hypot(*(np.asarray(b["center"][:2]) - mid)) < 0.06
    assert n_boxes > 150          # the rules don't reject almost everything
    assert n_between > 10         # the interesting case actually happens


def test_keep_clear_spots_stay_uncovered_and_visible(scene):
    """A box on (or hiding) the calibration AprilTag is rejected; sampled layouts never do either."""
    s, cam, top = scene
    tag = s["obstacles"]["keep_clear"][0]
    on_tag = {"id": "block", "size": [0.1, 0.1, 0.1], "center": [tag["xy"][0], tag["xy"][1], top + 0.05], "yaw": 0.0}
    probs = layout_problems([on_tag], [0.45, 0.25], [0.45, -0.3], cam, s["table"], top, 0.1018, s["obstacles"])
    assert any("keep-clear" in p for p in probs)
    rng = np.random.default_rng(1)
    for _ in range(100):
        boxes = sample_layout(rng, s["obstacles"], [0.5, 0.2], [0.5, -0.3], cam, s["table"], top, 0.1018)
        assert all(footprint_distance(tag["xy"], b) >= tag["r_m"] for b in boxes)


def test_too_close_boxes_stand_beside_the_target():
    """obstacles.too_close_prob: one box with its face too_close_gap_m from the target's side (the case to refuse);
    every other rule still holds for the layout."""
    s = load_yaml("config/scenes/so101_workcell.yaml")
    from pnp_core.catalog import load_fixed_camera
    cam = load_fixed_camera(s)["mount"]["position"]
    t = s["table"]
    top = t["center"][2] + t["size"][2] / 2
    cfg = {**s["obstacles"], "too_close_prob": 1.0}
    rng = np.random.default_rng(5)
    can, goal, half = (0.19, 0.05), (0.17, -0.07), 0.0079
    near_seen = 0
    for _ in range(20):
        boxes = sample_layout(rng, cfg, can, goal, cam, t, top, 0.0113, target_half_m=half)
        near = [b for b in boxes if b.get("near_target")]
        assert len(near) <= 1
        if near:
            near_seen += 1
            lo, hi = cfg["too_close_gap_m"]
            assert lo - 1e-6 <= footprint_distance(can, near[0]) - half <= hi + 1e-6
        assert not layout_problems(boxes, can, goal, cam, t, top, 0.0113, cfg)
    assert near_seen >= 10                                   # most layouts manage to put one beside it
