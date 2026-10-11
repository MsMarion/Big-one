"""What a real stereo depth camera does to a perfect depth image (config/sim_reality.yaml, docs/learning/15_sim_realism.md).

What this file teaches: a simulated depth image is exact; a real one is not, and the errors have STRUCTURE that
matters more than their size:
  axial noise     grows with distance squared (stereo: depth from disparity; a fixed sub-pixel disparity error is a
                  depth error ~ z^2 / (focal * baseline)). D435i at 1 m: ~2.6 mm; D405 at 25 cm: ~0.8 mm.
  correlated      part of it is smooth over a patch (stereo matching windows): a flat table comes out wavy, which
                  per-pixel averaging does not remove
  holes           blobs without depth (no texture, shiny, black)
  edges           at a depth jump the matcher fails (no depth) or blends both surfaces (a "flying pixel" floating in
                  the air between object and table: in an obstacle map that becomes a phantom obstacle)
  range           nothing closer than min_m (stereo can't match), nothing reliable beyond max_m
  scale error     a systematic % error of every depth (per camera unit)
Pure numpy (no ROS), deterministic for a given rng: unit-tested in tests/test_sensor_noise.py.
"""

from __future__ import annotations

import numpy as np


_POOL: dict = {}


def _white(rng: np.random.Generator, shape) -> np.ndarray:
    """Standard-normal noise of `shape` (float32), FAST: one of 8 pre-generated fields, rolled by a random offset.
    Drawing ~300k fresh normals per frame cost ~10 ms per image in Python and slowed 4 camera streams from 15 to
    ~8 Hz; the pool keeps the statistics (each frame: a random field at a random shift) at ~1 ms."""
    shape = tuple(shape)
    if shape not in _POOL:
        _POOL[shape] = np.random.default_rng(12345).standard_normal((8,) + shape, dtype=np.float32)
    f = _POOL[shape][int(rng.integers(8))]
    return np.roll(f, (int(rng.integers(shape[0])), int(rng.integers(shape[1]))), axis=(0, 1))


def _upsample(a: np.ndarray, shape, patch: int) -> np.ndarray:
    """Nearest-neighbor blow-up of a coarse grid to `shape` (blocky on purpose: cheap, and the stereo blobs are)."""
    out = np.repeat(np.repeat(a, patch, axis=0), patch, axis=1)
    return out[:shape[0], :shape[1]]


def depth_jumps(z: np.ndarray, jump: float) -> np.ndarray:
    """Pixels next to a depth discontinuity larger than `jump` (to any 4-neighbor)."""
    e = np.zeros(z.shape, bool)
    dx = np.abs(np.diff(z, axis=1)) > jump
    dy = np.abs(np.diff(z, axis=0)) > jump
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:, :] |= dy
    e[:-1, :] |= dy
    return e


def noisy_depth(z: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    """Perfect metric depth (0 / nan / inf = no return) -> realistic metric depth (0 = no return)."""
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0).astype(np.float32)
    k = float(p.get("k", 0.0))
    if k <= 0.0 and not p.get("holes") and not p.get("scale_error"):
        return z.astype(np.float32)
    valid = z > 0
    h, w = z.shape
    patch = int(p.get("patch_px", 16))
    c = float(p.get("correlated", 0.0))
    coarse = rng.standard_normal((h // patch + 1, w // patch + 1)).astype(np.float32)
    noise = np.float32(np.sqrt(1 - c)) * _white(rng, z.shape) + np.float32(np.sqrt(c)) * _upsample(coarse, z.shape, patch)
    out = z * np.float32(1.0 + float(p.get("scale_error", 0.0))) + np.float32(k) * z * z * noise
    # edges: dropout or flying pixels (a blend of the two surfaces)
    jump = float(p.get("edge_jump_m", 0.03))
    edges = depth_jumps(z, jump) & valid
    drop = np.zeros(z.shape, bool)
    fly = np.zeros(z.shape, bool)
    ei = np.flatnonzero(edges)                              # random draws only where there are edges
    if ei.size:
        u = rng.random(ei.size)
        pd, pf = float(p.get("edge_dropout", 0.0)), float(p.get("flying", 0.0))
        drop.flat[ei[u < pd]] = True
        fly.flat[ei[(u >= pd) & (u < pd + pf)]] = True
    if fly.any():
        far = np.maximum.reduce([np.roll(z, s, a) for a in (0, 1) for s in (-2, 2)])
        out[fly] = out[fly] + rng.random(int(fly.sum())).astype(np.float32) * (far[fly] - z[fly])
    # holes: random blobs
    holes = float(p.get("holes", 0.0))
    if holes > 0:
        blob = _upsample(rng.random((h // 8 + 1, w // 8 + 1)) < holes, z.shape, 8)
        drop |= blob
    out[drop | ~valid] = 0.0
    out[(out < float(p.get("min_m", 0.0))) | (out > float(p.get("max_m", np.inf)))] = 0.0
    return out.astype(np.float32)


def noisy_color(rgb: np.ndarray, p: dict, rng: np.random.Generator) -> np.ndarray:
    s = float(p.get("color_sigma", 0.0))
    if s <= 0:
        return rgb
    n = (_white(rng, rgb.shape) * np.float32(s)).astype(np.int16)       # pooled (see _white)
    return np.clip(rgb.astype(np.int16) + n, 0, 255).astype(np.uint8)
