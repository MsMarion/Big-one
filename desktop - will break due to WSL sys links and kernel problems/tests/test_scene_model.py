import numpy as np
import pytest

from pnp_core.scene_model import cluster_objects, compare_boxes, fit_box


def box_surface_points(center_xy, size, yaw, table_z=0.0, n=1500, seed=0):
    """Points on the top face and two side faces of a box (what a camera above-and-in-front sees)."""
    rng = np.random.default_rng(seed)
    sx, sy, sz = size
    top = np.column_stack([rng.uniform(-sx / 2, sx / 2, n), rng.uniform(-sy / 2, sy / 2, n), np.full(n, sz)])
    side = np.column_stack([np.full(n // 2, sx / 2), rng.uniform(-sy / 2, sy / 2, n // 2), rng.uniform(0, sz, n // 2)])
    p = np.vstack([top, side])
    c, s = np.cos(yaw), np.sin(yaw)
    p[:, :2] = p[:, :2] @ np.array([[c, s], [-s, c]]) + np.asarray(center_xy)
    p[:, 2] += table_z
    return p


def test_fit_box_recovers_size_and_pose():
    p = box_surface_points([0.6, -0.1], [0.26, 0.03, 0.16], np.radians(30))
    b = fit_box(p, 0.0)
    assert b["size"][0] == pytest.approx(0.26, abs=0.005)
    assert b["size"][1] == pytest.approx(0.03, abs=0.005)
    assert b["size"][2] == pytest.approx(0.16, abs=0.005)
    assert np.allclose(b["center"][:2], [0.6, -0.1], atol=0.005)
    assert np.degrees(b["yaw"]) == pytest.approx(30, abs=2)


def test_cluster_two_objects_and_compare():
    a = box_surface_points([0.5, 0.2], [0.10, 0.10, 0.10], 0.3, seed=1)
    b = box_surface_points([0.7, -0.2], [0.06, 0.06, 0.24], 0.0, seed=2)
    clusters = cluster_objects(np.vstack([a, b]))
    assert len(clusters) == 2
    measured = [fit_box(c, 0.0) for c in clusters]
    truth = [{"id": "block", "center": [0.5, 0.2, 0.05], "size": [0.10, 0.10, 0.10], "yaw": 0.3},
             {"id": "tower", "center": [0.7, -0.2, 0.12], "size": [0.06, 0.06, 0.24], "yaw": 0.0},
             {"id": "ghost", "center": [0.0, 0.0, 0.1], "size": [0.1, 0.1, 0.1], "yaw": 0.0}]
    rep = {r["id"]: r for r in compare_boxes(measured, truth)}
    assert rep["block"]["matched"] and rep["tower"]["matched"] and not rep["ghost"]["matched"]
    assert max(abs(v) for v in rep["block"]["size_err_mm"]) < 6
    assert abs(rep["block"]["yaw_err_deg"]) < 3
    assert rep["tower"]["center_err_mm"] < 5
