"""The desk flat map (pnp_core/height_map.py) and the scan view choice (pnp_core/wrist_scan.py), synthetic data."""
import numpy as np

from pnp_core.catalog import load_yaml
from pnp_core.geometry import make_transform, rot_z
from pnp_core.height_map import HeightMap, classify
from pnp_core.reorient import rest_poses
from pnp_core.wrist_scan import choose_views, table_footprint

CAT = load_yaml("config/objects.yaml")


def box_points(center, L, W, H, yaw, step=0.001, rng=None, noise=0.0):
    """Points of a box's top face + the desk around it (what a camera from above sees)."""
    rng = rng or np.random.default_rng(0)
    xs, ys = np.meshgrid(np.arange(-0.04, 0.04, step), np.arange(-0.04, 0.04, step))
    local = np.column_stack([xs.ravel(), ys.ravel()])
    inside = (np.abs(local[:, 0]) <= L / 2) & (np.abs(local[:, 1]) <= W / 2)
    R = rot_z(yaw)[:2, :2]
    xy = local @ R.T + center
    z = np.where(inside, H, 0.0) + rng.normal(0, noise, len(xy))
    return np.column_stack([xy, z]), inside


def test_a_brick_comes_out_with_its_size_yaw_and_identity():
    hm = HeightMap((0.10, 0.30), (-0.10, 0.10), res=0.002)
    pts, inside = box_points(np.array([0.19, 0.03]), 0.0318, 0.0158, 0.0113, 0.4, noise=0.0005)
    colors = np.where(inside[:, None], [[30, 30, 200]], [[200, 200, 200]]).astype(np.uint8)
    hm.add(pts, colors)
    objs = hm.objects()
    assert len(objs) == 1
    o = objs[0]
    assert np.allclose(o.center_xy, (0.19, 0.03), atol=0.002)
    assert abs(o.footprint[0] - 0.0318) < 0.004 and abs(o.footprint[1] - 0.0158) < 0.004
    assert abs(o.height - 0.0113) < 0.0015
    assert abs((o.yaw - 0.4 + np.pi / 2) % np.pi - np.pi / 2) < np.radians(6)
    m = classify(o, CAT, ["lego_2x4", "lego_2x2", "lego_1x2"], rest_poses)
    assert m["object"] == "lego_2x4" and m["rest"] in ("upright", "upside_down")
    assert hm.render(objects=objs).shape[2] == 3


def test_every_catalog_brick_is_told_apart_by_size():
    """The real desk's bricks (2x2, 2x3, 2x4), all catalog bricks as candidates (as wrist_map_node does)."""
    bricks = [k for k, v in CAT.items() if "brick" in v]
    for name, (L, W) in {"lego_2x4": (0.0318, 0.0158), "lego_2x3": (0.0238, 0.0158), "lego_2x2": (0.0158, 0.0158)}.items():
        hm = HeightMap((0.10, 0.30), (-0.10, 0.10), res=0.002)
        pts, _ = box_points(np.array([0.2, 0.0]), L, W, 0.0113, 0.7, noise=0.0005)
        hm.add(pts)
        objs = hm.objects()
        assert len(objs) == 1, name
        m = classify(objs[0], CAT, bricks, rest_poses)
        assert m["object"] == name and m["rest"] in ("upright", "upside_down"), (name, m, objs[0].footprint)


def test_flying_pixels_do_not_make_walls():
    hm = HeightMap((0.0, 0.1), (0.0, 0.1), res=0.002, min_points=3)
    pts = np.column_stack([np.random.default_rng(1).uniform(0, 0.1, (5000, 2)), np.zeros(5000)])
    spikes = np.array([[0.05, 0.05, 0.03], [0.02, 0.07, 0.05]])       # single bad pixels
    hm.add(np.vstack([pts, spikes]))
    assert hm.objects() == []


def test_a_standing_brick_is_recognised_on_its_end():
    hm = HeightMap((0.10, 0.30), (-0.10, 0.10), res=0.002)
    pts, _ = box_points(np.array([0.2, 0.0]), 0.0158, 0.0113, 0.0318, 0.0)    # 2x4 standing on its short end
    hm.add(pts)
    m = classify(hm.objects()[0], CAT, ["lego_2x4", "lego_2x2", "lego_1x2"], rest_poses)
    assert m["object"] == "lego_2x4" and m["rest"].startswith("side_x")


def test_table_footprint_of_a_camera_looking_straight_down():
    K = np.array([[331.0, 0, 320], [0, 331.0, 180], [0, 0, 1]])
    T = make_transform(np.diag([1.0, -1.0, -1.0]), [0.2, 0.0, 0.30])          # optical z down, 30 cm up
    poly = table_footprint(T, K, (640, 360), 0.0)
    assert np.allclose(poly.mean(axis=0), [0.2, 0.0], atol=1e-6)
    assert abs((poly[:, 0].max() - poly[:, 0].min()) - 0.30 * 640 / 331.0) < 1e-6


def test_choose_views_covers_the_region_with_few_views():
    K = np.array([[331.0, 0, 320], [0, 331.0, 180], [0, 0, 1]])

    def fk_cam(q):                     # a toy "arm": camera above (q0, q1), straight down, height q2
        return make_transform(np.diag([1.0, -1.0, -1.0]), [q[0], q[1], q[2]])
    views, cov = choose_views(fk_cam, [0.1, -0.15, 0.15], [0.3, 0.15, 0.3], (0.12, 0.28, -0.12, 0.12), K,
                              (640, 360), 0.0, np.random.default_rng(0), n_samples=2000)
    assert cov > 0.95 and 1 <= len(views) <= 8


def test_a_tilted_offset_desk_is_levelled_and_the_brick_keeps_its_height():
    """An uncalibrated wrist camera: the desk 20 mm low and tilted 6 deg. The fit finds it; the brick stays 11.3 mm."""
    from pnp_core.height_map import fit_desk_plane, level_to_plane
    pts, inside = box_points(np.array([0.2, 0.0]), 0.0318, 0.0158, 0.0113, 0.3, noise=0.0005)
    a = np.radians(6)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    c = np.array([0.2, 0.0, 0.0])
    bad = (pts - c) @ R.T + c + [0, 0, -0.020]
    fit = fit_desk_plane(bad, z_hint=0.0)
    assert fit is not None
    n, d, frac = fit
    assert abs(np.degrees(np.arccos(n[2])) - 6) < 0.5 and frac > 0.6
    hm = HeightMap((0.10, 0.30), (-0.10, 0.10), res=0.002)
    hm.add(level_to_plane(bad, n, d, 0.0))
    objs = hm.objects()
    assert len(objs) == 1 and abs(objs[0].height - 0.0113) < 0.0015


def test_holes_in_a_shiny_top_do_not_make_the_height_nan():
    """The real white brick: some top cells get no depth. The blob still has its height and identity."""
    # 0.5 mm point spacing ~ the D405 at 20 cm (fx ~330 px: ~3 px per 2 mm cell, ~11 points per cell)
    pts, inside = box_points(np.array([0.2, 0.0]), 0.0318, 0.0158, 0.0113, 0.0, step=0.0005, noise=0.0003)
    rng = np.random.default_rng(1)
    holes = inside & (np.abs(pts[:, 0] - 0.2) < 0.002) & (np.abs(pts[:, 1]) < 0.002)      # a small bare spot mid-top
    holes |= inside & (rng.random(len(pts)) < 0.15)
    hm = HeightMap((0.10, 0.30), (-0.10, 0.10), res=0.002)
    hm.add(pts[~holes])
    objs = hm.objects()
    assert len(objs) == 1 and np.isfinite(objs[0].height) and abs(objs[0].height - 0.0113) < 0.0015
    bricks = [k for k, v in CAT.items() if "brick" in v]
    assert classify(objs[0], CAT, bricks, rest_poses)["object"] == "lego_2x4"


def test_a_side_view_wall_does_not_inflate_the_outline():
    """One oblique view: the near side wall's points (0..H) lie just outside the top. The outline stays the top's."""
    pts, inside = box_points(np.array([0.2, 0.0]), 0.0318, 0.0158, 0.0113, 0.0, step=0.0005, noise=0.0003)
    wall = []
    for z in np.arange(0.0005, 0.0113, 0.0005):                     # the wall facing y-, smeared 0..4 mm outward
        for x in np.arange(0.2 - 0.0159, 0.2 + 0.0159, 0.0005):
            wall.append([x, -0.0079 - 0.004 * (1 - z / 0.0113), z])
    hm = HeightMap((0.10, 0.30), (-0.10, 0.10), res=0.002)
    hm.add(np.vstack([pts, np.array(wall)]))
    o = hm.objects()[0]
    assert abs(o.footprint[0] - 0.0318) < 0.004 and abs(o.footprint[1] - 0.0158) < 0.004, o.footprint
    assert abs(o.center_xy[1]) < 0.002
