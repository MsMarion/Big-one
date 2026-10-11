"""Depth image -> obstacle point cloud for MoveIt's Octomap (Phase 5, Layer B: obstacles the robot SEES).

Pipeline (obstacle_cloud(); the ROS wrapper is pnp_perception/obstacle_cloud_node.py):

  depth (every `step`-th pixel) -> 3D points (camera) -> base frame -> crop to the workspace
    -> drop the ROBOT (a capsule around each link's URDF collision geometry: it sees its own arm!)
    -> drop the target CANS (from the detector: we want to grasp them, not avoid them)
    -> split: table surface (kept only so MoveIt can clear free space along the rays) | things ON the table
    -> things on the table: voxel-downsample, then "columnize" (fill down to the table)

Why columnize (2.5-D assumption): a camera sees only the faces of a box turned towards it. Octomap treats unseen
space as FREE, so the arm could plan straight through the hidden back half of a box. On a tabletop, objects
stand on the table (no overhangs), so every point seen at height h is assumed to be solid down to the table.
That turns the visible top face into a full block. (Wrong for shelves/arches; fine for boxes on a table.)

Pure numpy; unit-tested in tests/test_cloud.py.
"""

from __future__ import annotations

import numpy as np


def depth_to_cloud(depth_m: np.ndarray, fx, fy, cx, cy, step: int = 4, min_d: float = 0.15,
                   max_d: float = 2.0) -> np.ndarray:
    """Every `step`-th pixel of a metric depth image -> (N,3) points in the camera OPTICAL frame."""
    d = depth_m[::step, ::step]
    vs, us = np.mgrid[0:depth_m.shape[0]:step, 0:depth_m.shape[1]:step]
    ok = (d > min_d) & (d < max_d)
    z = d[ok].astype(np.float64)
    return np.column_stack([(us[ok] - cx) * z / fx, (vs[ok] - cy) * z / fy, z])


def drop_depth_edges(depth_m: np.ndarray, jump: float, grow: int = 1) -> np.ndarray:
    """Zero the pixels at depth discontinuities (> jump to a 4-neighbor), grown by `grow` pixels.
    A real stereo camera's 'flying pixels' sit exactly there: values between the two surfaces, i.e. points floating
    in the air beside every object and beside the moving arm (seen with the realistic sensor profile: phantom
    voxels next to fr3_link6 invalidated the arm's motion). RealSense pipelines filter them the same way."""
    if jump <= 0:
        return depth_m
    d = depth_m
    e = np.zeros(d.shape, bool)
    dx = np.abs(np.diff(d, axis=1)) > jump
    dy = np.abs(np.diff(d, axis=0)) > jump
    e[:, 1:] |= dx
    e[:, :-1] |= dx
    e[1:, :] |= dy
    e[:-1, :] |= dy
    for _ in range(grow):
        g = e.copy()
        g[1:, :] |= e[:-1, :]
        g[:-1, :] |= e[1:, :]
        g[:, 1:] |= e[:, :-1]
        g[:, :-1] |= e[:, 1:]
        e = g
    out = d.copy()
    out[e] = 0.0
    return out


def far_rays(depth_m: np.ndarray, fx, fy, cx, cy, step: int = 4, min_d: float = 2.0, max_d: float = 20.0) -> np.ndarray:
    """Pixels that see something FARTHER than the mapping range (valid depth in [min_d, max_d)) -> their viewing
    rays as camera-frame points at z = 1 (push them beyond the Octomap range: free space, nothing marked).
    Why: a voxel in the air is only cleared by a ray that passes through it to something behind it. From our
    camera, the rays through the space above the far table edge end on the floor ~3.4 m away; dropping those
    returns (depth_to_cloud stops at 2 m) left stale voxels there forever (a removed tower's top above the goal).
    Zero depth (no return, holes on a real D435i: shiny/black surfaces) is NOT treated as free."""
    d = depth_m[::step, ::step]
    vs, us = np.mgrid[0:depth_m.shape[0]:step, 0:depth_m.shape[1]:step]
    ok = (d >= min_d) & (d < max_d)
    return np.column_stack([(us[ok] - cx) / fx, (vs[ok] - cy) / fy, np.ones(int(ok.sum()))])


def segment_distance(points: np.ndarray, a, b) -> np.ndarray:
    """Distance of each point to the segment a-b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip((points - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def near_capsules(points: np.ndarray, chain, radius: float) -> np.ndarray:
    """Mask of points within `radius` of the polyline through `chain` (robot link origins, base -> tool)."""
    m = np.zeros(len(points), bool)
    for a, b in zip(chain[:-1], chain[1:]):
        m |= segment_distance(points, a, b) < radius
    return m


def in_cylinders(points: np.ndarray, centers_xy, radius: float, z_max: float) -> np.ndarray:
    """Mask of points inside vertical cylinders (radius, from below up to z_max) around each center."""
    m = np.zeros(len(points), bool)
    for c in centers_xy:
        m |= (np.hypot(points[:, 0] - c[0], points[:, 1] - c[1]) < radius) & (points[:, 2] < z_max)
    return m


def voxel_downsample(points: np.ndarray, res: float, min_points: int = 1) -> np.ndarray:
    """One point (the voxel center) per occupied voxel of size `res`, counting only voxels hit by >= min_points.
    min_points > 1 removes depth-noise SPECKLE: a real surface puts several points into a 2 cm voxel (~9 at 1 m
    with every 4th pixel), a stray noise point is alone (seen with realistic noise: single voxels all over the table)."""
    if len(points) == 0:
        return points
    keys, counts = np.unique(np.floor(points / res).astype(np.int64), axis=0, return_counts=True)
    return (keys[counts >= min_points] + 0.5) * res


def columnize(points: np.ndarray, floor_z: float, res: float) -> np.ndarray:
    """Add points every `res` from each point straight down to floor_z (input should be voxel-downsampled)."""
    cols = [points]
    for p in points:
        zs = np.arange(p[2] - res, floor_z, -res)
        if len(zs):
            cols.append(np.column_stack([np.full(len(zs), p[0]), np.full(len(zs), p[1]), zs]))
    return np.vstack(cols) if len(points) else points


def obstacle_cloud(points_base: np.ndarray, table_z: float, workspace: dict, robot_capsules, robot_margin: float,
                   cans_xy, can_radius: float, can_height: float, res: float = 0.02,
                   surface_band: float = 0.015, min_points: int = 1, cut_out: list | None = None,
                   cut_from_cloud: bool = True, table_points: list | None = None):
    """Base-frame points -> (cloud for the Octomap, removed points, object points), all base frame.
    object points = what stands on the table, before voxels/columns, the targets excluded (for measuring:
    pnp_core.scene_model). See the module docstring.

    robot_capsules: [(a, b, radius)] in the base frame, one per robot link (fit_capsule on the URDF collision
    geometry), possibly for two instants (the image's time AND now); points within radius + robot_margin go.
    The REMOVED points (robot, cans) are returned so the caller can still use them to CLEAR space: a point that
    is simply dropped casts no ray, so a voxel inserted earlier at that spot (e.g. the can before the detector
    saw it) would stay forever.
    cut_from_cloud: True = the target cylinders (cans_xy) are cut out of the Octomap cloud (the old way: it also
    cut whatever stood beside the target). False = the cloud keeps everything; the pick node puts the target into
    MoveIt as a collision object at its pose and MoveIt filters exactly that shape. The cylinders then only keep the
    targets out of the object points (the scene model, the change monitor).
    cut_out: a list -> the points the cylinders removed FROM THE CLOUD are appended to it (to show the cut-out).
    table_points: a list -> everything standing on the table (robot removed, NO target cut) is appended to it: what
    a pickability check tests the gripper against."""
    p = points_base
    x, y = workspace["x"], workspace["y"]
    keep = ((p[:, 0] > x[0]) & (p[:, 0] < x[1]) & (p[:, 1] > y[0]) & (p[:, 1] < y[1])
            & (p[:, 2] > table_z - 0.03) & (p[:, 2] < table_z + workspace["z_max"]))
    # Points OUTSIDE the workspace (the floor past the table edge, walls) are not obstacles, but their rays are the
    # only ones crossing the air high above the far side of the table: dropping them left stale voxels there
    # forever (seen: the top of a removed 24 cm tower stayed in the Octomap above the goal). -> removed (clear only).
    outside, p = p[~keep], p[keep]
    drop = np.zeros(len(p), bool)
    if robot_capsules:
        drop |= near_capsule_list(p, robot_capsules, robot_margin)
    cyl = (in_cylinders(p, cans_xy, can_radius, table_z + can_height + 0.03) if len(cans_xy)
           else np.zeros(len(p), bool))
    if table_points is not None:
        table_points.append(p[~drop & (p[:, 2] > table_z + surface_band)])
    if cut_from_cloud:
        if cut_out is not None:
            cut_out.append(p[cyl & ~drop])
        drop |= cyl
    target = cyl[~drop]                                     # target points still in the cloud (cut_from_cloud=False)
    removed, p = np.vstack([outside, p[drop]]), p[~drop]
    on_table = p[:, 2] > table_z + surface_band
    surface = voxel_downsample(p[~on_table], res)          # free-space rays only (MoveIt drops them: table object)
    objects = columnize(voxel_downsample(p[on_table], res, min_points), table_z + res / 2, res)
    cloud = np.vstack([surface, objects]) if len(surface) or len(objects) else np.zeros((0, 3))
    return cloud, voxel_downsample(removed, res), p[on_table & ~target]


def push_beyond(points_cam: np.ndarray, distance: float) -> np.ndarray:
    """Move camera-frame points along their viewing ray to `distance` from the camera. The Octomap updater treats
    points beyond its max_range as 'free all the way, no obstacle at the end': clearing without marking."""
    n = np.linalg.norm(points_cam, axis=1, keepdims=True)
    return points_cam / np.maximum(n, 1e-9) * distance


# --- robot self-filter from the URDF collision geometry ------------------------------------------------------
def load_stl_vertices(path) -> np.ndarray:
    """Vertices (N,3) of a binary or ASCII STL file (duplicates included; fine for fitting)."""
    data = open(path, "rb").read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = [line.split()[1:4] for line in data.decode(errors="ignore").splitlines() if line.strip().startswith("vertex")]
        return np.array(v, float)
    n = int(np.frombuffer(data[80:84], np.uint32)[0])
    rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rec["v"].reshape(-1, 3).astype(np.float64)


def box_vertices(size, xyz=(0, 0, 0), R=np.eye(3)) -> np.ndarray:
    h = np.asarray(size, float) / 2
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    return corners @ np.asarray(R).T + np.asarray(xyz, float)


def fit_capsule(vertices: np.ndarray):
    """Smallest-ish capsule (a, b, radius) around a vertex set: axis = principal direction (PCA), radius = largest
    distance of a vertex to that axis, ends = extreme projections pulled in by the radius (the caps cover them)."""
    c = vertices.mean(axis=0)
    _, _, vt = np.linalg.svd(vertices - c, full_matrices=False)
    axis = vt[0]
    t = (vertices - c) @ axis
    radial = np.linalg.norm((vertices - c) - np.outer(t, axis), axis=1)
    r = float(radial.max())
    t0, t1 = t.min() + r, t.max() - r
    if t0 > t1:                                          # short and fat: degenerate to a sphere
        t0 = t1 = (t.min() + t.max()) / 2
        r = float(np.linalg.norm(vertices - (c + t0 * axis), axis=1).max())
    return c + t0 * axis, c + t1 * axis, r


def near_capsule_list(points: np.ndarray, capsules, margin: float) -> np.ndarray:
    """Mask of points within (radius + margin) of any capsule (a, b, radius), all in the same frame."""
    m = np.zeros(len(points), bool)
    for a, b, r in capsules:
        m |= segment_distance(points, a, b) < r + margin
    return m
