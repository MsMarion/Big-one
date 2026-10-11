"""The plate's AprilTag layout (config/calibration_targets_so101_plate.yaml, tools/tag_layout.py): every tag fully in
the fixed camera's view at the plate scene's camera pose, the printed squares on the board and apart from each other."""
import itertools
import sys

import pytest

sys.path.insert(0, "/repo/tools")
tag_layout = pytest.importorskip("tag_layout")
from pnp_core.catalog import load_yaml  # noqa: E402

SCENE = "so101_plate_tags"


@pytest.fixture(scope="module")
def layout():
    scene, _ = tag_layout.scene_of(SCENE)
    rows, a, geom = tag_layout.visibility(scene)
    return scene, rows, a


def test_every_tag_seen_and_big_enough(layout):
    _, rows, _ = layout
    assert len(rows) >= 6
    assert all(r["seen"] for r in rows), [r["id"] for r in rows if not r["seen"]]
    assert min(r["px"] for r in rows) >= 35             # px across the black square: corners need ~6 px per cell


def test_tags_on_the_board_and_apart(layout):
    scene, rows, a = layout
    half = a["size_m"] * 1.5 / 2                         # the printed square incl. its white border
    t = scene["table"]
    x0, x1 = t["center"][0] - t["size"][0] / 2, t["center"][0] + t["size"][0] / 2
    y0, y1 = t["center"][1] - t["size"][1] / 2, t["center"][1] + t["size"][1] / 2
    for r in rows:
        assert x0 <= r["x"] - half and r["x"] + half <= x1 and y0 <= r["y"] - half and r["y"] + half <= y1, r["id"]
    for p, q in itertools.combinations(rows, 2):
        gap = max(abs(p["x"] - q["x"]), abs(p["y"] - q["y"])) - 2 * half
        assert gap >= 0.01, (p["id"], q["id"], gap)


def test_ids_unique_and_outside_the_pick_area(layout):
    scene, rows, _ = layout
    assert len({r["id"] for r in rows}) == len(rows)
    rg = scene["target"]["spawn_region"]
    for r in rows:
        inside = rg["x"][0] <= r["x"] <= rg["x"][1] and rg["y"][0] <= r["y"] <= rg["y"][1]
        assert not inside, r["id"]


def test_hand_and_table_boards_match_the_desk_cell():
    plate = load_yaml("config/calibration_targets_so101_plate.yaml")
    desk = load_yaml("config/calibration_targets_so101.yaml")
    assert plate["hand_board"] == desk["hand_board"] and plate["table_board"] == desk["table_board"]
