"""The staged vision pipeline (pnp_core/vision.py) on a synthetic RGB-D frame: bricks on a desk, seen from above."""
import numpy as np

from pnp_core.catalog import load_yaml
from pnp_core.vision import Frame, build_pipeline

CAT = load_yaml("config/objects.yaml")
TABLE_Z = 0.0
REGION = (0.10, 0.30, -0.12, 0.12)


def render(bricks, cam_xy=(0.2, 0.0), cam_h=0.25, size=(640, 480), f=600.0):
    """Depth of a camera looking straight down (image up = base +x) at boxes lying on the desk.
    bricks: [(name, (x, y), yaw)]. -> Frame."""
    w, h = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])     # optical x, y, z in the base frame
    T[:3, 3] = [cam_xy[0], cam_xy[1], TABLE_Z + cam_h]
    vs, us = np.mgrid[0:h, 0:w]
    rx, ry = (us - K[0, 2]) / f, (vs - K[1, 2]) / f                          # ray per unit depth (optical)
    depth = np.full((h, w), cam_h)
    for name, (bx, by), yaw in bricks:
        L, W, H = CAT[name]["dims_m"]
        d = cam_h - H                                                        # depth of the brick's top face
        x = T[0, 3] + (T[:3, :3] @ np.stack([rx * d, ry * d, np.full_like(rx, d)]).reshape(3, -1))[0].reshape(h, w)
        y = T[1, 3] + (T[:3, :3] @ np.stack([rx * d, ry * d, np.full_like(rx, d)]).reshape(3, -1))[1].reshape(h, w)
        c, s = np.cos(yaw), np.sin(yaw)
        along, across = (x - bx) * c + (y - by) * s, -(x - bx) * s + (y - by) * c
        top = (np.abs(along) <= L / 2) & (np.abs(across) <= W / 2)
        depth[top] = np.minimum(depth[top], d)
    return Frame(None, depth, K, T)


def test_geometry_profile_finds_and_names_the_bricks():
    cfg = load_yaml("config/vision.yaml")["profiles"]["geometry"]
    pipe = build_pipeline(cfg, CAT, TABLE_Z, REGION)
    truth = {"lego_2x4": (0.19, 0.03, 0.4), "lego_2x2": (0.23, -0.05, 0.0)}
    est = pipe.run(render([(n, (x, y), yaw) for n, (x, y, yaw) in truth.items()]))
    got = {e.identity.name: e for e in est}
    assert set(got) == set(truth), [e.identity.name for e in est]
    for name, (x, y, _yaw) in truth.items():
        e = got[name]
        assert np.hypot(e.T_base_obj[0, 3] - x, e.T_base_obj[1, 3] - y) < 0.002, (name, e.T_base_obj[:3, 3])
        assert abs(e.T_base_obj[2, 3] - CAT[name]["dims_m"][2] / 2) < 1e-6
        assert e.pose_source == "box_geometry" and e.identity.source == "catalog_size"
        assert e.proposal.mask.sum() > 100                                   # its pixels are marked in the image


def test_unknown_backend_is_a_clear_error():
    import pytest
    with pytest.raises(ValueError, match="no identity backend 'nonexistent'"):
        build_pipeline({"identity": [{"name": "nonexistent"}]}, CAT, TABLE_Z, REGION)


def test_crop_for_embedding_keeps_only_the_masked_pixels():
    from pnp_core.vision import crop_for_embedding
    rgb = np.zeros((100, 120, 3), np.uint8)
    rgb[40:60, 50:90] = (200, 30, 30)
    rgb[0:10, 0:10] = (0, 255, 0)                                          # outside the mask: must not appear
    mask = np.zeros((100, 120), bool)
    mask[40:60, 50:90] = True
    c = crop_for_embedding(rgb, mask)
    assert c.shape[0] == c.shape[1] and c.shape[0] >= 40
    assert (c == (200, 30, 30)).all(axis=2).sum() == 20 * 40
    assert not (c == (0, 255, 0)).all(axis=2).any()
