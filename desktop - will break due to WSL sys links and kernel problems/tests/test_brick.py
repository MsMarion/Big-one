"""pnp_core.brick: the brick mesh FoundationPose gets has the catalog's size, the sim's frame, outward faces."""
import numpy as np
import pytest

from pnp_core.brick import brick_geometry, brick_mesh
from pnp_core.catalog import load_yaml

BRICKS = {k: v for k, v in load_yaml("config/objects.yaml").items() if "brick" in v}


@pytest.mark.parametrize("name", sorted(BRICKS))
def test_mesh_matches_the_catalog_and_is_centered(name):
    V, F = brick_mesh(BRICKS[name])
    assert np.allclose(V.max(0) - V.min(0), BRICKS[name]["dims_m"], atol=1e-6)
    assert np.allclose((V.max(0) + V.min(0)) / 2, 0.0, atol=1e-6)          # origin = bounding-box center (sim prim)
    assert F.min() == 0 and F.max() == len(V) - 1


@pytest.mark.parametrize("name", sorted(BRICKS))
def test_faces_point_outward(name):
    """Every triangle's normal points away from the part it belongs to (body center or stud axis)."""
    V, F = brick_mesh(BRICKS[name])
    g = brick_geometry(BRICKS[name])
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    c = V[F].mean(axis=1)
    top_of_body = g.body_center[2] + g.body_size[2] / 2
    ref = np.tile(np.asarray(g.body_center, float), (len(F), 1))
    on_stud = c[:, 2] > top_of_body + 1e-9
    studs = np.asarray(g.stud_centers)
    nearest = studs[np.argmin(np.linalg.norm(c[on_stud, None, :2] - studs[None, :, :2], axis=2), axis=1)]
    ref[on_stud] = nearest
    out = c - ref
    cap = on_stud & (np.abs(n[:, 2]) > 0.99 * np.linalg.norm(n, axis=1))     # a stud's top: "out" = straight up
    out[cap, :2] = 0
    assert np.all(np.einsum("ij,ij->i", n, out) > 0)
