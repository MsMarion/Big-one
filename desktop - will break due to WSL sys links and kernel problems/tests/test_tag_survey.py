"""pnp_core.calibration.survey_tags: hand-placed tags (cm off the plan) -> their real layout + the camera, from one
image. Synthetic: a D435i-like view of 9 tags displaced by ~1 cm / 3 deg each."""

import cv2
import numpy as np

from pnp_core.calibration import flat_tag_T, pose_error, survey_tags, tag_corners
from pnp_core.geometry import make_transform

PLANNED = {1: (-0.03, 0.22, 0), 2: (-0.045, -0.27, 0), 3: (0.0, 0.12, 0), 4: (0.0, -0.13, 0), 5: (0.11, 0.185, 0),
           6: (0.11, -0.215, 0), 7: (0.30, 0.125, 0), 8: (0.30, -0.15, 0), 9: (0.33, 0.0, 0)}
K = np.array([[466.0, 0, 320], [0, 466.0, 240], [0, 0, 1]])          # the D435i colour: 69 deg across 640 px


def camera():
    """base -> optical: at (0.53, 0.01, 0.33), looking back at the robot ~50 deg down."""
    p, at = np.array([0.53, 0.01, 0.33]), np.array([0.08, 0.0, 0.0])
    z = (at - p) / np.linalg.norm(at - p)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return make_transform(np.column_stack([x, np.cross(z, x), z]), p)


def test_survey_recovers_relative_layout_and_camera():
    rng = np.random.default_rng(3)
    X = camera()
    # careful taping (~1.5 mm, ~1 deg) + ONE tag put 2 cm off (no room at its spot): the survey must single it out
    true = {i: (x + rng.normal(0, 0.0015), y + rng.normal(0, 0.0015), yaw + rng.normal(0, 1)) for i, (x, y, yaw) in
            PLANNED.items()}
    true[7] = (true[7][0] - 0.02, true[7][1] + 0.005, true[7][2])
    Xi = np.linalg.inv(X)
    obs = {}
    for i, (x, y, yaw) in true.items():
        T = Xi @ flat_tag_T((x, y), yaw, 0.0)
        uv, _ = cv2.projectPoints(tag_corners(0.0612) @ T[:3, :3].T + T[:3, 3], np.zeros(3), np.zeros(3), K, None)
        obs[i] = uv.reshape(-1, 2) + rng.normal(0, 0.15, (4, 2))
        assert np.all((obs[i] > 0) & (obs[i] < [640, 480])), f"tag {i} out of the synthetic view"
    X0 = X @ make_transform(np.eye(3), [0.01, -0.01, 0.005])                # a start a cm off
    Xs, lay, rms, size = survey_tags(obs, 0.06, K, None, PLANNED, 0.0, X0)
    assert abs(size - 0.0612) < 0.0015      # the corners were rendered 2 % out (blur): roughly found
    assert rms < 0.4
    off = {i: np.hypot(lay[i][0] - PLANNED[i][0], lay[i][1] - PLANNED[i][1]) * 1000 for i in PLANNED}
    assert off[7] > 15, off                                # the misplaced tag stands out
    # the others: read as placed (vs the truth, the set's placement taken out: the check compares against the plan)
    true_off = {i: np.hypot(true[i][0] - PLANNED[i][0], true[i][1] - PLANNED[i][1]) * 1000 for i in PLANNED}
    assert max(abs(off[i] - true_off[i]) for i in PLANNED if i != 7) < 2.0, (off, true_off)
    e = pose_error(Xs, X)
    assert e[0] < 15 and e[1] < 1.0, e
