"""pnp_core/pickability.py: the open hand's space vs the points on the table, before any motion."""
import numpy as np
import pytest

from pnp_core.catalog import load_robot, load_yaml
from pnp_core.gripper import GripperModel
from pnp_core.object_grasp import upright_pose
from pnp_core.pickability import gripper_volumes, pick_check

TABLE_Z = -0.0024


@pytest.fixture(scope="module")
def so101():
    cfg = load_robot("so101")
    return cfg, GripperModel(cfg["gripper"]), load_yaml("config/objects.yaml")


def cube_points(cx, cy, size=0.03, step=0.002):
    """The visible surface of a cube standing on the table (top + 4 sides), as a depth camera would roughly see."""
    h = size / 2
    a = np.arange(-h, h + 1e-9, step)
    top = np.array([[cx + x, cy + y, TABLE_Z + size] for x in a for y in a])
    zs = np.arange(TABLE_Z + 0.006, TABLE_Z + size, step)
    sides = [[cx + s * h, cy + u, z] for s in (-1, 1) for u in a for z in zs] + \
            [[cx + u, cy + s * h, z] for s in (-1, 1) for u in a for z in zs]
    return np.vstack([top, np.array(sides)])


def brick_points(obj, T):
    ext = np.asarray(obj["dims_m"], float)
    g = np.mgrid[-0.5:0.51:0.1, -0.5:0.51:0.1].reshape(2, -1).T
    top = np.column_stack([g * ext[:2], np.full(len(g), ext[2] / 2)])
    return top @ T[:3, :3].T + T[:3, 3]


def check(so101, gap, side=(0.0, 1.0), yaw=0.0, name="lego_2x2"):
    cfg, gm, cat = so101
    obj = cat[name]
    T = upright_pose(obj, (0.17, 0.0), TABLE_Z + obj["dims_m"][2] / 2, yaw)
    d = np.asarray(side, float) / np.linalg.norm(side)
    # the cube's near face `gap` beyond the brick's side (measured along d from the brick's half-extent)
    half = abs(d @ (T[:2, :2] @ (np.asarray(obj["dims_m"][:2]) / 2 * np.sign(T[:2, :2].T @ d))))
    c = np.array([0.17, 0.0]) + d * (half + gap + 0.015)
    pts = np.vstack([brick_points(obj, T), cube_points(*c)])
    return pick_check(pts, obj, T, cfg["gripper"], gm, yaws=(yaw, yaw + np.pi / 2),
                      approach_height=cfg["grasp_motion"]["approach_height_m"], label=name.replace("_", " "),
                      inflate=0.01)                          # the SO-101 cell's 1 cm voxels


def test_volumes_of_the_so101(so101):
    cfg, _, _ = so101
    names = {v[0] for v in gripper_volumes(cfg["gripper"], 0.032, 0.04)}
    assert names == {"fixed jaw", "open jaw", "gripper body"}     # from gripper.hand_bands (the meshes)


def test_nothing_near_is_pickable(so101):
    cfg, gm, cat = so101
    obj = cat["lego_2x2"]
    T = upright_pose(obj, (0.17, 0.0), TABLE_Z + obj["dims_m"][2] / 2, 0.3)
    r = pick_check(brick_points(obj, T), obj, T, cfg["gripper"], gm, yaws=(0.3,))
    assert r["pickable"] and r["why"] == ""                   # its own points never block it


def test_a_cube_touching_blocks_every_grasp(so101):
    r = check(so101, gap=0.002)
    assert not r["pickable"]
    assert "obstacle" in r["why"] and "needs" in r["why"]


def test_a_cube_far_away_does_not_block(so101):
    assert check(so101, gap=0.055)["pickable"]


@pytest.mark.parametrize("side", [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)])
def test_close_on_any_side_blocks_some_grasp(so101, side):
    r = check(so101, gap=0.005, side=side)
    assert len(r["clear"]) < len(r["results"])               # at least the grasp whose jaw is on that side is blocked


def test_the_reason_measures_along_the_obstacle(so101):
    r = check(so101, gap=0.002)
    gap = float(r["why"].split("obstacle ")[1].split(" cm")[0])
    need = float(r["why"].split("needs ")[1].split(" cm")[0])
    assert gap == pytest.approx(0.2, abs=0.15)            # the cube's near face, 2 mm from the brick's side
    assert need > gap                                     # the jaw reaches further than the gap


def test_check_inflate_is_voxel_plus_margin():
    from pnp_core.pickability import check_inflate
    assert check_inflate({}) == pytest.approx(0.02)
    sc = {"perception": {"obstacle_cloud": {"voxel_m": 0.01}, "pickability_margin_m": 0.004}}
    assert check_inflate(sc) == pytest.approx(0.014)


def test_grasp_yaws_shared_by_page_and_pick():
    import math
    from pnp_core.grasp import grasp_yaws
    assert grasp_yaws(0.0, 0.1, False) == [0.0]                       # a box: one yaw, the object decides
    ys = grasp_yaws(0.0, 0.3, True)                                   # a can: the wrist's yaw first, then turned
    assert len(ys) == 4 and ys[0] == pytest.approx(0.3)
    assert all(abs(y) <= math.pi for y in ys)
