import numpy as np
import pytest

from pnp_core.geometry import transform_points
from pnp_core.perception_math import (
    Intrinsics,
    depth_to_meters,
    deproject,
    estimate_box,
    estimate_cylinder,
    mask_to_points,
)
from synthetic import render_cylinder

K = Intrinsics(615.0, 615.0, 320.0, 240.0)
W, H = 640, 480
EYE = [1.10, 0.50, 0.65]          # config/camera_d435i.yaml mount
TARGET = [0.50, 0.05, 0.0]
R_CAN, H_CAN = 0.033, 0.101       # tomato soup can


def test_depth_to_meters():
    mm = np.array([[0, 1000, 65535]], np.uint16)
    assert np.allclose(depth_to_meters(mm, "16UC1"), [[0.0, 1.0, 65.535]])
    f = np.array([[np.nan, 0.5, np.inf]], np.float32)
    assert np.allclose(depth_to_meters(f, "32FC1"), [[0.0, 0.5, 0.0]])
    with pytest.raises(ValueError):
        depth_to_meters(mm, "rgb8")


def test_project_deproject_roundtrip():
    pts = np.array([[0.1, -0.05, 0.8], [-0.2, 0.1, 1.5]])
    uv = K.project(pts)
    assert np.allclose(deproject(uv[:, 0], uv[:, 1], pts[:, 2], K), pts)


@pytest.mark.parametrize("center", [(0.50, 0.20), (0.62, 0.08), (0.42, 0.33)])
def test_cylinder_center_from_rendered_depth(center):
    depth, mask, T_base_cam = render_cylinder(K, W, H, EYE, TARGET, center, R_CAN, H_CAN)
    assert mask.sum() > 500, "can should be clearly visible from the configured mount"

    pts_cam = mask_to_points(mask, depth, K)
    pts_base = transform_points(T_base_cam, pts_cam)

    naive_err = np.linalg.norm(pts_base[:, :2].mean(axis=0) - center)
    est = estimate_cylinder(pts_base, R_CAN, camera_xy=EYE[:2])
    err = np.linalg.norm(est.center_xy - center)

    assert err < 0.003, f"center error {err * 1000:.1f} mm"
    assert naive_err > 3 * err, "shape fit should beat the naive centroid (camera-side bias)"
    assert abs(est.z_top - H_CAN) < 0.005
    assert est.fit_residual_m < 0.002


def test_cylinder_robust_to_depth_noise():
    center = (0.50, 0.20)
    depth, mask, T = render_cylinder(K, W, H, EYE, TARGET, center, R_CAN, H_CAN)
    rng = np.random.default_rng(0)
    noisy = depth * (1 + rng.normal(0, 0.002, depth.shape)).astype(np.float32)   # ~2 mm at 1 m
    holes = rng.random(depth.shape) < 0.05                                       # 5% dropouts
    noisy[holes] = 0.0
    est = estimate_cylinder(transform_points(T, mask_to_points(mask, noisy, K)), R_CAN, camera_xy=EYE[:2])
    assert np.linalg.norm(est.center_xy - center) < 0.006


def test_empty_mask_gives_no_points():
    assert mask_to_points(np.zeros((H, W), bool), np.ones((H, W), np.float32), K).shape == (0, 3)


def test_estimate_object_end_to_end():
    from pnp_core.catalog import load_object
    from pnp_core.perception_math import estimate_object

    can = load_object("tomato_soup_can")
    center = (0.55, 0.15)
    depth, mask, T = render_cylinder(K, W, H, EYE, TARGET, center, can["dims_m"][0] / 2, can["dims_m"][2])
    est, why = estimate_object(mask, depth, K, T, can)
    assert est is not None, why
    assert np.linalg.norm(est.center_xy - center) < 0.003
    assert abs(est.z_top - can["dims_m"][2]) < 0.005


def test_estimate_object_rejects_wrong_shapes():
    from pnp_core.catalog import load_object
    from pnp_core.perception_math import estimate_object

    can = load_object("tomato_soup_can")
    # a mask over plain table (e.g. a false detection): points are flat at z=0 -> wrong top height
    depth, mask, T = render_cylinder(K, W, H, EYE, TARGET, (0.55, 0.15), 0.033, 0.101)
    table_only = np.zeros_like(mask)
    table_only[300:340, 100:160] = True
    est, why = estimate_object(table_only, depth, K, T, can)
    assert est is None and ("height" in why or "residual" in why), why
    # a taller cylinder (another can type): top height check rejects it
    depth, mask, T = render_cylinder(K, W, H, EYE, TARGET, (0.55, 0.15), 0.051, 0.140)
    est, why = estimate_object(mask, depth, K, T, can)
    assert est is None, "a 140 mm tall can must not be accepted as the 102 mm soup can"


def test_box_yaw():
    rng = np.random.default_rng(1)
    yaw = 0.4
    local = rng.uniform([-0.08, -0.03, 0.0], [0.08, 0.03, 0.2], size=(2000, 3))   # long side along x
    c, s = np.cos(yaw), np.sin(yaw)
    pts = local @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [0.5, 0.1, 0.0]
    est = estimate_box(pts)
    assert np.allclose(est.center_xy, [0.5, 0.1], atol=0.005)
    assert abs(est.yaw - yaw) < np.deg2rad(2)


def test_mask_erosion_adapts_to_small_masks():
    # a 1x2 LEGO brick from 48 cm: ~15 x 8 px. A fixed 3 px erosion leaves a 9 x 2 strip; the adaptive one keeps
    # most of it, and a big mask is still eroded by the full 3 px
    small = np.zeros((H, W), bool)
    small[100:108, 100:115] = True
    depth = np.full((H, W), 0.48, np.float32)
    assert len(mask_to_points(small, depth, K, erode_px=3)) <= 18
    assert len(mask_to_points(small, depth, K)) >= 60
    big = np.zeros((H, W), bool)
    big[100:160, 100:160] = True
    assert len(mask_to_points(big, depth, K)) == len(mask_to_points(big, depth, K, erode_px=3))
