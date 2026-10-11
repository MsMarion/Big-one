"""LEGO brick geometry, one source for the sim (sim/scene.py add_brick) and the FoundationPose meshes
(tools/make_brick_meshes.py): a box body + cylindrical studs on top, origin at the center of the whole bounding box
incl. studs (= objects.yaml dims_m, like the YCB assets), z up.

LEGO dimensions: 8 mm stud pitch, 0.1 mm play per side, 4.8 mm stud diameter, 1.7 mm stud height; the body height
is dims_m z minus the studs (9.6 mm for a brick).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PITCH, PLAY, STUD_D, STUD_H = 0.008, 0.0001, 0.0048, 0.0017


@dataclass
class BrickGeometry:
    body_size: tuple          # (L, W, H) of the body box
    body_center: tuple        # its center (the studs sit on its top)
    stud_centers: list        # (x, y, z) of each stud's center
    stud_radius: float
    stud_height: float


def brick_geometry(obj_cfg: dict) -> BrickGeometry:
    nx, ny = obj_cfg["brick"]["studs"]
    L, W, H = nx * PITCH - 2 * PLAY, ny * PITCH - 2 * PLAY, obj_cfg["dims_m"][2] - STUD_H
    studs = [((i - (nx - 1) / 2) * PITCH, (j - (ny - 1) / 2) * PITCH, H / 2) for i in range(nx) for j in range(ny)]
    return BrickGeometry((L, W, H), (0.0, 0.0, -STUD_H / 2), studs, STUD_D / 2, STUD_H)


def brick_mesh(obj_cfg: dict, segments: int = 24):
    """Triangle mesh of the brick (outward-facing, counter-clockwise): -> (vertices (N, 3), faces (M, 3))."""
    g = brick_geometry(obj_cfg)
    V, F = [], []

    def add(verts, faces):
        base = len(V)
        V.extend(verts)
        F.extend([[a + base, b + base, c + base] for a, b, c in faces])

    # body: 8 corners, 12 triangles
    (L, W, H), (cx, cy, cz) = g.body_size, g.body_center
    corners = [(cx + sx * L / 2, cy + sy * W / 2, cz + sz * H / 2) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    add(corners, [[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6],          # bottom, top
                  [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7],          # -y, +y
                  [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])         # -x, +x
    # studs: side wall + top cap (the bottom sits on the body's top face)
    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([np.cos(a), np.sin(a)]) * g.stud_radius
    for x, y, z in g.stud_centers:
        z0, z1 = z - g.stud_height / 2, z + g.stud_height / 2
        verts = [(x + u, y + v, z0) for u, v in ring] + [(x + u, y + v, z1) for u, v in ring] + [(x, y, z1)]
        n, top_c = segments, 2 * segments
        faces = []
        for k in range(n):
            k2 = (k + 1) % n
            faces += [[k, k2, n + k], [k2, n + k2, n + k], [n + k, n + k2, top_c]]
        add(verts, faces)
    return np.asarray(V, float), np.asarray(F, int)


def linear_to_srgb(c):
    """Linear color (what the sim's material takes) -> sRGB (what a camera image shows)."""
    c = np.clip(np.asarray(c, float), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
