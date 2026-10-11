"""Change monitor for a FROZEN obstacle map (map_mode: freeze_per_task, docs/learning/13_map_modes.md).

What this file teaches: when the robot plans a whole task against a snapshot of the scene (frozen map), it must
notice if the real scene changes. Two questions, answered from each live depth frame:

  APPEARED  something is occupied now that was free in the snapshot:
            live obstacle voxels that are not within `dilation` voxels of any snapshot voxel
            AND that the snapshot's own depth images saw as empty space (seen through). A spot that was hidden at
            snapshot time (behind the arm) was UNKNOWN, not free: something there coming into view is not a change.
  GONE      something in the snapshot is not there any more. Absence can only be confirmed by LOOKING THROUGH it:
            project each snapshot voxel into the depth image; if the camera measures a surface clearly BEHIND the
            voxel, the line of sight passes through empty space -> gone. If the measured surface is IN FRONT of the
            voxel (the arm, another object), we simply can't see it: unknown, never "gone".

Small clusters are noise (the arm filter's edges, depth speckle): only clusters of >= min_voxels count, and the
node additionally requires a change to persist for a few frames.
"""

from __future__ import annotations

import numpy as np

N26 = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


def voxel_keys(points: np.ndarray, res: float) -> set:
    """(N,3) points -> set of integer voxel keys."""
    if len(points) == 0:
        return set()
    return set(map(tuple, np.floor(np.asarray(points) / res).astype(np.int64)))


def key_centers(keys, res: float) -> np.ndarray:
    return (np.array(sorted(keys), float) + 0.5) * res if keys else np.zeros((0, 3))


def dilate(keys: set, r: int = 1) -> set:
    out = set(keys)
    rng = range(-r, r + 1)
    for k in keys:
        for dx in rng:
            for dy in rng:
                for dz in rng:
                    out.add((k[0] + dx, k[1] + dy, k[2] + dz))
    return out


def clusters(keys: set, min_voxels: int) -> list:
    """Connected components (26-neighborhood) with at least min_voxels voxels, largest first."""
    left, out = set(keys), []
    while left:
        seed = left.pop()
        comp, stack = [seed], [seed]
        while stack:
            k = stack.pop()
            for d in N26:
                n = (k[0] + d[0], k[1] + d[1], k[2] + d[2])
                if n in left:
                    left.remove(n)
                    comp.append(n)
                    stack.append(n)
        if len(comp) >= min_voxels:
            out.append(comp)
    return sorted(out, key=len, reverse=True)


def newly_seen(live: set, snapshot: set, observed_free, min_voxels: int = 10, dilation: int = 1) -> list:
    """Clusters of live voxels the snapshot could NOT have known: not in it, and not seen as empty either (hidden,
    e.g. behind the arm at home). Not a change, but missing knowledge: the map must take them in before planning.
    (Found: ignoring them let the robot plan through a wall it could not see at snapshot time, 2 collisions.)"""
    cand = live - dilate(snapshot, dilation)
    return clusters(cand - observed_free(cand), min_voxels)


def appeared(live: set, snapshot: set, min_voxels: int = 10, dilation: int = 1, observed_free=None) -> list:
    """Clusters of live voxels that the snapshot doesn't explain.
    observed_free(candidates) -> the subset the SNAPSHOT saw as empty space (seen through). Without it, everything
    that was merely hidden at snapshot time (e.g. behind the arm at home) and comes into view later looks "new":
    newly VISIBLE is not the same as newly THERE (found: real boxes reported as 'appeared' in 3 of 20 tasks)."""
    cand = live - dilate(snapshot, dilation)
    if observed_free is not None:
        cand = observed_free(cand)
    return clusters(cand, min_voxels)


def seen_through(snapshot: set, res: float, depth_m: np.ndarray, fx, fy, cx, cy, T_cam_base: np.ndarray,
                 margin: float = 0.04) -> set:
    """Snapshot voxels whose line of sight now ends clearly BEHIND them (camera sees through = empty now).
    depth_m: metric depth image (0 = no return: unknown); T_cam_base: base -> camera optical frame.

    The WHOLE projected footprint of the voxel must see through, not just its center pixel: a voxel on the edge of
    an object that is still there has its center just outside the surface, and the ray past the edge sees the
    table behind (found by the unit test: a present box looked partly "gone")."""
    if not snapshot:
        return set()
    keys = sorted(snapshot)
    c = (np.array(keys, float) + 0.5) * res
    p = c @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    z = p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.round(fx * p[:, 0] / z + cx).astype(np.int64)
        v = np.round(fy * p[:, 1] / z + cy).astype(np.int64)
    h, w = depth_m.shape
    ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    gone = set()
    for i in np.flatnonzero(ok):
        r = int(0.5 * fx * res / z[i]) + 1                     # half the voxel's size in pixels (+1)
        patch = depth_m[max(0, v[i] - r):v[i] + r + 1, max(0, u[i] - r):u[i] + r + 1]
        if patch.size and patch.min() > z[i] + margin:         # any 0 (no return) or nearer surface -> not gone
            gone.add(keys[i])
    return gone


def summarize(comps: list, res: float) -> list:
    """Clusters -> [{"center": [x, y, z], "voxels": n, "size": [dx, dy, dz]}] (base frame, meters)."""
    out = []
    for comp in comps:
        k = np.array(comp, float)
        lo, hi = k.min(axis=0) * res, (k.max(axis=0) + 1) * res
        out.append({"center": np.round((lo + hi) / 2, 3).tolist(), "voxels": len(comp),
                    "size": np.round(hi - lo, 3).tolist()})
    return out
