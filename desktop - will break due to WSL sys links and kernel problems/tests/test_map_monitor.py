import numpy as np

from pnp_core.geometry import look_at_optical, make_transform
from pnp_core.map_monitor import appeared, clusters, dilate, newly_seen, seen_through, summarize, voxel_keys

RES = 0.02


def box_points(center, size, step=0.01):
    c, s = np.asarray(center, float), np.asarray(size, float)
    axes = [np.arange(c[i] - s[i] / 2, c[i] + s[i] / 2 + 1e-9, step) for i in range(3)]
    return np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T


def test_voxel_keys_and_dilate():
    k = voxel_keys(np.array([[0.011, 0.0, 0.0], [0.019, 0.001, 0.0]]), RES)
    assert k == {(0, 0, 0)}
    assert len(dilate(k, 1)) == 27


def test_clusters_split_and_size_filter():
    a = voxel_keys(box_points([0.5, 0.0, 0.05], [0.06, 0.06, 0.1]), RES)
    b = voxel_keys(np.array([[0.9, 0.4, 0.05]]), RES)                       # a single speckle voxel
    comps = clusters(a | b, min_voxels=5)
    assert len(comps) == 1 and len(comps[0]) == len(a)


def test_a_new_box_appears_but_the_snapshot_does_not():
    snap = voxel_keys(box_points([0.5, 0.0, 0.05], [0.06, 0.06, 0.1]), RES)
    live = snap | voxel_keys(box_points([0.4, -0.3, 0.06], [0.08, 0.08, 0.12]), RES)
    comps = appeared(live, snap, min_voxels=10)
    assert len(comps) == 1
    s = summarize(comps, RES)[0]
    assert np.allclose(s["center"][:2], [0.4, -0.3], atol=0.02)
    assert appeared(snap, snap) == []


def test_jitter_by_one_voxel_is_not_a_change():
    snap = voxel_keys(box_points([0.5, 0.0, 0.05], [0.06, 0.06, 0.1]), RES)
    live = voxel_keys(box_points([0.51, 0.01, 0.05], [0.06, 0.06, 0.1]), RES)
    assert appeared(live, snap, min_voxels=3, dilation=1) == []


def _camera():
    eye, target = np.array([1.1, 0.5, 0.65]), np.array([0.5, 0.05, 0.0])
    T_base_cam = make_transform(look_at_optical(eye, target), eye)
    return np.linalg.inv(T_base_cam), (600.0, 600.0, 320.0, 240.0)


def _render(points_base, T_cam_base, K, shape=(480, 640)):
    """A depth image of these points (nearest surface per pixel)."""
    fx, fy, cx, cy = K
    d = np.full(shape, np.inf)
    p = points_base @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    u = np.round(fx * p[:, 0] / p[:, 2] + cx).astype(int)
    v = np.round(fy * p[:, 1] / p[:, 2] + cy).astype(int)
    ok = (u >= 0) & (u < shape[1]) & (v >= 0) & (v < shape[0])
    for uu, vv, zz in zip(u[ok], v[ok], p[ok, 2]):           # 3x3 splat: no pinholes between sparse points
        sl = (slice(max(0, vv - 1), vv + 2), slice(max(0, uu - 1), uu + 2))
        d[sl] = np.minimum(d[sl], zz)
    d[np.isinf(d)] = 0.0
    return d


def test_a_removed_box_is_seen_through_but_an_occluded_one_is_unknown():
    T_cam_base, K = _camera()
    table = box_points([0.45, 0.0, 0.0], [0.9, 1.2, 0.0], step=0.004)
    snap = voxel_keys(box_points([0.5, -0.3, 0.06], [0.06, 0.06, 0.12]), RES)
    # the box is gone: the camera sees the table behind/below it
    gone = seen_through(snap, RES, _render(table, T_cam_base, K), *K, T_cam_base)
    assert len(clusters(gone, 5)) == 1
    # the box is still there: nothing is seen through
    still = np.vstack([table, box_points([0.5, -0.3, 0.06], [0.06, 0.06, 0.12], step=0.004)])
    assert len(clusters(seen_through(snap, RES, _render(still, T_cam_base, K), *K, T_cam_base), 5)) == 0
    # the box is gone but a big plate between the camera and it hides that: unknown, not "gone"
    plate = box_points([0.8, 0.1, 0.35], [0.3, 0.4, 0.0], step=0.004)
    hidden = _render(np.vstack([table, plate]), T_cam_base, K)
    assert len(clusters(seen_through(snap, RES, hidden, *K, T_cam_base), 5)) == 0


def test_newly_visible_is_not_newly_there():
    """A box hidden (behind a plate) when the snapshot was taken, visible later: not an 'appeared' change.
    The same box placed where the snapshot saw empty table: a change."""
    T_cam_base, K = _camera()
    table = box_points([0.45, 0.0, 0.0], [0.9, 1.2, 0.0], step=0.004)
    box = box_points([0.5, -0.3, 0.06], [0.06, 0.06, 0.12], step=0.004)
    plate = box_points([0.8, 0.1, 0.35], [0.3, 0.4, 0.0], step=0.004)            # hides the box's spot
    live = voxel_keys(box_points([0.5, -0.3, 0.06], [0.06, 0.06, 0.12]), RES)
    for snap_scene, expect in ((np.vstack([table, box, plate]), 0),               # hidden at snapshot time
                               (table, 1)):                                        # snapshot saw empty table
        snap_depth = _render(snap_scene, T_cam_base, K)
        free = lambda keys, d=snap_depth: seen_through(keys, RES, d, *K, T_cam_base)   # noqa: E731
        assert len(appeared(live, set(), min_voxels=5, observed_free=free)) == expect


def test_hidden_at_snapshot_time_is_newly_seen():
    T_cam_base, K = _camera()
    table = box_points([0.45, 0.0, 0.0], [0.9, 1.2, 0.0], step=0.004)
    box = box_points([0.5, -0.3, 0.06], [0.06, 0.06, 0.12], step=0.004)
    plate = box_points([0.8, 0.1, 0.35], [0.3, 0.4, 0.0], step=0.004)
    live = voxel_keys(box_points([0.5, -0.3, 0.06], [0.06, 0.06, 0.12]), RES)
    d = _render(np.vstack([table, box, plate]), T_cam_base, K)                   # snapshot: hidden behind the plate
    free = lambda keys: seen_through(keys, RES, d, *K, T_cam_base)               # noqa: E731
    assert len(newly_seen(live, set(), free, min_voxels=5)) == 1
