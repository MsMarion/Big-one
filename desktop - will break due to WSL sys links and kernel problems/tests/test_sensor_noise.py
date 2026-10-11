import numpy as np

from pnp_core.catalog import load_yaml
from pnp_core.sensor_noise import depth_jumps, noisy_color, noisy_depth

REAL = load_yaml("config/sim_reality.yaml")["profiles"]["realistic"]


def flat(z, shape=(240, 320)):
    return np.full(shape, z, np.float32)


def test_ideal_is_untouched():
    z = flat(0.8)
    assert np.array_equal(noisy_depth(z, {"k": 0.0}, np.random.default_rng(0)), z)


def test_axial_noise_grows_with_distance_squared():
    p = {"k": 0.0026}
    s1 = noisy_depth(flat(0.5), p, np.random.default_rng(1)).std()
    s2 = noisy_depth(flat(1.0), p, np.random.default_rng(1)).std()
    assert 3.5 < s2 / s1 < 4.5
    assert abs(s2 - 0.0026) < 0.0003                       # D435i at 1 m: ~2.6 mm


def test_correlated_noise_is_smooth_over_patches():
    d = noisy_depth(flat(1.0), {"k": 0.0026, "correlated": 0.9, "patch_px": 16}, np.random.default_rng(2)) - 1.0
    r = np.corrcoef(d[:, :-1].ravel(), d[:, 1:].ravel())[0, 1]
    assert r > 0.6                                         # neighbors move together (white noise: ~0)


def test_range_limits_and_scale_error():
    z = np.concatenate([flat(0.2, (10, 10)), flat(1.0, (10, 10)), flat(3.5, (10, 10))])
    out = noisy_depth(z, {"k": 1e-9, "min_m": 0.28, "max_m": 3.0, "scale_error": 0.01}, np.random.default_rng(3))
    assert (out[:10] == 0).all() and (out[20:] == 0).all()        # too close / too far: no return
    assert np.allclose(out[10:20], 1.01, atol=1e-5)                # 1 % scale error


def test_depth_edges_drop_or_fly():
    z = flat(1.0)
    z[:, 160:] = 0.6                                       # an object edge: 40 cm depth jump
    assert depth_jumps(z, 0.03)[:, 159:161].all()
    out = noisy_depth(z, {"k": 1e-9, "edge_dropout": 0.5, "flying": 0.5, "edge_jump_m": 0.03},
                      np.random.default_rng(4))
    edge = out[:, 159:161]
    assert (edge == 0).mean() > 0.3                        # dropped
    between = (edge > 0.62) & (edge < 0.98)
    assert between.any()                                   # flying pixels between the two surfaces
    assert np.allclose(out[:, :150], 1.0, atol=1e-5)       # away from the edge: untouched


def test_realistic_profile_runs_and_keeps_most_pixels():
    z = flat(0.9, (480, 640))
    out = noisy_depth(z, REAL["d435i"], np.random.default_rng(5))
    assert 0.95 < (out > 0).mean() <= 1.0
    rgb = np.full((48, 64, 3), 128, np.uint8)
    assert noisy_color(rgb, REAL["d435i"], np.random.default_rng(6)).std() > 1.0
