"""pnp_core/hand_detect.py: the sim stand-in's magenta blobs, pixels -> base frame, the workspace test."""

import numpy as np

from pnp_core.hand_detect import in_workspace, magenta_blobs, pixels_to_base


def test_magenta_blob_found_and_others_ignored():
    img = np.full((120, 160, 3), 128, np.uint8)
    img[20:50, 30:80] = (242, 13, 217)          # the prop's colour
    img[70:100, 100:150] = (200, 40, 40)        # red brick: not a hand
    img[5:8, 5:8] = (242, 13, 217)              # a speck: below min_area
    blobs = magenta_blobs(img, min_area_px=50)
    assert len(blobs) == 1
    u, v = blobs[0].mean(axis=0)
    assert abs(u - 54.5) < 1 and abs(v - 34.5) < 1


def test_pixels_to_base_and_workspace():
    K = np.array([[300.0, 0, 80], [0, 300.0, 60], [0, 0, 1]])
    T = np.eye(4)                                # camera = base, looking along +z
    depth = np.full((120, 160), 0.5, np.float32)
    p = pixels_to_base(np.array([[80, 60], [81, 60], [80, 61], [79, 59]]), depth, K, T)
    assert np.allclose(p, [0.0, 0.0, 0.5], atol=2e-3)
    assert pixels_to_base(np.array([[80, 60]]), np.zeros_like(depth), K, T) is None
    assert in_workspace(np.array([0.2, 0.1, 0.1]), 0.0, 0.45, 0.45)
    assert not in_workspace(np.array([0.6, 0.0, 0.1]), 0.0, 0.45, 0.45)      # beyond the reach
    assert not in_workspace(np.array([0.2, 0.0, 0.8]), 0.0, 0.45, 0.45)      # high above the table
    assert in_workspace(None, 0.0, 0.45, 0.45)                               # no depth: pause to be safe
