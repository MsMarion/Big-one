import numpy as np
import pytest

from pnp_core.cloud import (columnize, depth_to_cloud, drop_depth_edges, far_rays, fit_capsule, in_cylinders, load_stl_vertices, near_capsule_list,
                            near_capsules, obstacle_cloud, push_beyond,
                            segment_distance, voxel_downsample)

WS = {"x": [-0.3, 1.0], "y": [-0.6, 0.6], "z_max": 0.6}


def test_depth_to_cloud_center_pixel():
    d = np.zeros((480, 640), np.float32)
    d[240, 320] = 1.0
    d[0, 0] = 5.0                                                # beyond max_d
    pts = depth_to_cloud(d, 615, 615, 320, 240, step=1)
    assert pts.shape == (1, 3)
    assert np.allclose(pts[0], [0, 0, 1.0])


def test_segment_distance_and_capsules():
    pts = np.array([[0.5, 0.1, 0.0], [1.5, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    assert np.allclose(segment_distance(pts, [0, 0, 0], [1, 0, 0]), [0.1, 0.5, 0.2])
    assert near_capsules(pts, [[0, 0, 0], [1, 0, 0]], 0.15).tolist() == [True, False, False]


def test_in_cylinders():
    pts = np.array([[0.5, 0.2, 0.05], [0.5, 0.2, 0.30], [0.6, 0.2, 0.05]])
    assert in_cylinders(pts, [[0.5, 0.2]], 0.05, 0.13).tolist() == [True, False, False]


def test_voxel_and_columnize():
    pts = np.array([[0.101, 0.101, 0.101], [0.109, 0.105, 0.102], [0.3, 0.3, 0.05]])
    v = voxel_downsample(pts, 0.02)
    assert len(v) == 2
    col = columnize(np.array([[0.11, 0.11, 0.11]]), 0.01, 0.02)
    assert np.allclose(sorted(col[:, 2]), [0.03, 0.05, 0.07, 0.09, 0.11])
    assert np.allclose(col[:, :2], 0.11)


def test_obstacle_cloud_box_robot_can():
    """A box top surface becomes a solid column; robot and can points disappear; table points stay low."""
    rng = np.random.default_rng(0)
    box_top = np.column_stack([rng.uniform(0.6, 0.7, 300), rng.uniform(-0.1, 0.0, 300), np.full(300, 0.2)])
    arm = np.column_stack([np.full(100, 0.3), np.zeros(100), np.linspace(0.1, 0.5, 100)])
    can = np.column_stack([np.full(50, 0.5), np.full(50, 0.2), np.linspace(0.01, 0.10, 50)])
    table = np.column_stack([rng.uniform(0.2, 0.8, 200), rng.uniform(-0.3, 0.3, 200), np.zeros(200)])
    out, removed, objects = obstacle_cloud(np.vstack([box_top, arm, can, table]), 0.0, WS,
                                  robot_capsules=[([0.3, 0, 0.0], [0.3, 0, 0.6], 0.05)], robot_margin=0.03,
                                  cans_xy=[[0.5, 0.2]], can_radius=0.06, can_height=0.10)
    high = out[out[:, 2] > 0.015]
    assert len(high) > 0
    assert (high[:, 0] > 0.58).all() and (high[:, 0] < 0.72).all()      # only the box remains above the table
    assert high[:, 2].min() < 0.035 and high[:, 2].max() == pytest.approx(0.21, abs=0.011)   # solid to the table
    assert not (np.hypot(out[:, 0] - 0.5, out[:, 1] - 0.2) < 0.04).any()                    # can removed
    assert len(removed) > 0 and (np.hypot(removed[:, 0] - 0.3, removed[:, 1]) < 0.03).sum() > 0   # arm kept for clearing
    assert len(objects) == 300 and (objects[:, 2] == 0.2).all()                      # the box top, raw


def test_points_outside_the_workspace_are_kept_for_clearing():
    """The floor past the table edge is no obstacle, but its rays clear the air above the table's far side."""
    floor = np.column_stack([np.full(20, 1.4), np.linspace(-0.2, 0.2, 20), np.full(20, -0.75)])   # beyond x = 1.0
    out, removed, _ = obstacle_cloud(floor, 0.0, WS, robot_capsules=[], robot_margin=0.03,
                                     cans_xy=[], can_radius=0.06, can_height=0.10)
    assert len(out) == 0 and len(removed) > 0


def test_far_returns_become_clearing_rays_but_holes_do_not():
    d = np.zeros((480, 640), np.float32)
    d[240, 320] = 3.4            # the floor beyond the table edge: free space up to there
    d[100, 100] = 0.0            # no return (a hole on a real camera): unknown, not free
    d[300, 300] = 1.0            # within the mapping range: handled by depth_to_cloud
    rays = far_rays(d, 615, 615, 320, 240, step=1)
    assert rays.shape == (1, 3) and np.allclose(rays[0], [0, 0, 1])


def test_flying_pixels_at_depth_edges_are_dropped():
    d = np.full((20, 20), 1.0, np.float32)
    d[:, 10:] = 0.6                                       # an object edge
    d[:, 10] = 0.8                                        # a flying pixel column between the two surfaces
    out = drop_depth_edges(d, 0.03, grow=1)
    assert (out[:, 8:13] == 0).all()                      # the edge and its flying pixels are gone
    assert (out[:, :7] == 1.0).all() and (out[:, 14:] == 0.6).all()   # the surfaces themselves stay
    assert np.array_equal(drop_depth_edges(d, 0.0), d)   # 0 = off


def test_speckle_is_not_an_obstacle_but_a_surface_is():
    rng = np.random.default_rng(1)
    box_top = np.column_stack([rng.uniform(0.6, 0.7, 300), rng.uniform(-0.1, 0.0, 300), np.full(300, 0.2)])
    speckle = np.column_stack([rng.uniform(0.2, 0.8, 30), rng.uniform(-0.4, 0.4, 30), np.full(30, 0.025)])
    out, _, _ = obstacle_cloud(np.vstack([box_top, speckle]), 0.0, WS, robot_capsules=[], robot_margin=0.03,
                               cans_xy=[], can_radius=0.06, can_height=0.10, min_points=3)
    high = out[out[:, 2] > 0.015]
    assert len(high) and (high[:, 0] > 0.58).all() and (high[:, 0] < 0.72).all()   # only the box


def test_push_beyond():
    p = push_beyond(np.array([[0.0, 0.0, 1.0], [3.0, 4.0, 0.0]]), 2.5)
    assert np.allclose(p, [[0, 0, 2.5], [1.5, 2.0, 0.0]])


def test_fit_capsule_cylinder_like():
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 0.4, 2000)
    ang = rng.uniform(0, 2 * np.pi, 2000)
    pts = np.column_stack([0.05 * np.cos(ang) + 1.0, 0.05 * np.sin(ang), z])      # tube r=5 cm along z
    a, b, r = fit_capsule(pts)
    assert r == pytest.approx(0.05, abs=0.003)
    assert abs(abs(a[2] - b[2]) - 0.30) < 0.01                                       # 0.4 - 2r
    assert near_capsule_list(np.array([[1.0, 0.0, 0.2], [1.2, 0.0, 0.2]]), [(a, b, r)], 0.02).tolist() == [True, False]


def test_stl_roundtrip(tmp_path):
    tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    rec = np.zeros(1, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    rec["v"][0] = tri.ravel()
    f = tmp_path / "t.stl"
    f.write_bytes(b"\0" * 80 + np.uint32(1).tobytes() + rec.tobytes())
    assert np.allclose(load_stl_vertices(f), tri)


def test_target_left_in_the_cloud_keeps_a_neighbour_whole():
    """A 2x2 brick and a 3 cm cube 1 cm beside it. The old cut-out (a 23 mm cylinder around the brick) erased the
    cube's near side; with cut_from_cloud=False the cloud keeps both (MoveIt filters the brick as a collision object),
    the object points still leave the brick out, and table_points has everything."""
    g = np.mgrid[0:16, 0:16].reshape(2, -1).T
    brick = np.column_stack([0.165 - 0.0079 + g[:, 0] * 0.001, 0.009 - 0.0079 + g[:, 1] * 0.001, np.full(len(g), 0.009)])
    cube = np.column_stack([0.150 + np.tile(np.arange(30), 30) * 0.001, 0.028 + np.repeat(np.arange(30), 30) * 0.001,
                            np.full(900, 0.0276)])
    pts = np.vstack([brick, cube])
    ws = {"x": [-1, 1], "y": [-1, 1], "z_max": 0.7}
    args = (pts, -0.0024, ws, [], 0.0, [(0.165, 0.009)], 0.0232, 0.0113)
    cut = []
    obstacle_cloud(*args, res=0.01, surface_band=0.005, cut_out=cut)
    assert len(cut[0]) > len(brick)                       # the old cylinder took the brick AND part of the cube
    cut2, table2 = [], []
    cloud, _, objs = obstacle_cloud(*args, res=0.01, surface_band=0.005, cut_out=cut2, cut_from_cloud=False,
                                    table_points=table2)
    assert cut2 == []                                     # nothing cut from the cloud
    assert len(table2[0]) == len(pts)                     # the check sees everything
    near_cube = cloud[(np.hypot(cloud[:, 0] - 0.165, cloud[:, 1] - 0.009) < 0.0232) & (cloud[:, 1] > 0.022) & (cloud[:, 2] > 0.0)]
    assert len(near_cube) > 0                             # the cube's near side is in the map
    assert np.all(np.hypot(objs[:, 0] - 0.165, objs[:, 1] - 0.009) > 0.0232 - 1e-9)   # object points: brick left out
