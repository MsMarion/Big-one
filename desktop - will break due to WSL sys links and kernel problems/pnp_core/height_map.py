"""A flat map of the desk from depth images: a top-down height + color grid, and the objects on it
(docs/learning/21_wrist_map.md).

What this file teaches: with only a camera on the WRIST (no fixed camera over the desk), the robot builds its picture of
the desk by looking from a few poses and fusing what it sees. Each depth pixel becomes a 3-D point in the camera
frame; the arm's kinematics + the hand-eye calibration put it in the robot's base frame; the points fall into a grid
of 2 mm cells over the desk. Per cell we keep a ROBUST top height (not the highest point: single flying pixels at
object edges would make every edge a wall) and the color seen at that top. Objects are the connected blobs that
stand above the desk: their outline (minimum-area rectangle) gives center, footprint and yaw; the height and the
footprint say which catalog object it is and on which face it rests (pnp_core/reorient.py).

Pure numpy / OpenCV: tested with synthetic points (tests/test_height_map.py), used by the wrist_map_node.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class MapObject:
    center_xy: tuple
    footprint: tuple          # (long, short) [m]
    yaw: float                # long side direction, (-pi/2, pi/2]
    height: float             # top above the desk [m]
    color_bgr: tuple
    cells: int
    match: dict = field(default_factory=dict)      # {"object", "rest", "error"} from classify()


class HeightMap:
    def __init__(self, x_range, y_range, res: float = 0.002, table_z: float = 0.0, max_height: float = 0.10,
                 top_pct: float = 50.0, min_points: int = 3):
        self.x0, self.y0, self.res, self.table_z = float(x_range[0]), float(y_range[0]), float(res), float(table_z)
        self.nx = int(np.ceil((x_range[1] - x_range[0]) / res))
        self.ny = int(np.ceil((y_range[1] - y_range[0]) / res))
        self.max_height, self.top_pct, self.min_points = max_height, top_pct, min_points
        self.layers = []                                            # per frame: the cell heights it saw (nan: not seen)
        self._H = None
        self.C = np.zeros((self.ny, self.nx, 3), np.uint8)         # color at the top (BGR)
        self.seen = np.zeros((self.ny, self.nx), np.int32)          # frames that observed the cell

    def cell_of(self, xy):
        return ((np.asarray(xy)[..., 0] - self.x0) / self.res).astype(int), \
               ((np.asarray(xy)[..., 1] - self.y0) / self.res).astype(int)

    def center_of(self, ix, iy):
        return self.x0 + (np.asarray(ix) + 0.5) * self.res, self.y0 + (np.asarray(iy) + 0.5) * self.res

    def add(self, points: np.ndarray, colors_bgr: np.ndarray | None = None):
        """Fuse one frame's points (N, 3) in the base frame (+ their colors). Per cell of THIS frame: the top_pct
        percentile of its heights (needs min_points), then max over frames (each view sees other sides). The median
        (50): a cell on an object's edge counts as the object only if half its points are on it -- with 80 every
        rotated brick came out one cell (2 mm) bigger on each side."""
        p = np.asarray(points, float)
        h = p[:, 2] - self.table_z
        ix, iy = self.cell_of(p[:, :2])
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny) & (h > -0.01) & (h < self.max_height)
        if not ok.any():
            return 0
        ix, iy, h = ix[ok], iy[ok], h[ok]
        col = None if colors_bgr is None else np.asarray(colors_bgr, np.uint8)[ok]
        cell = iy * self.nx + ix
        order = np.lexsort((h, cell))                               # by cell, then height
        cell, h, ix, iy = cell[order], h[order], ix[order], iy[order]
        col = None if col is None else col[order]
        starts = np.flatnonzero(np.r_[True, cell[1:] != cell[:-1]])
        counts = np.diff(np.r_[starts, len(cell)])
        keep = counts >= self.min_points
        starts, counts = starts[keep], counts[keep]
        pick = starts + np.floor((counts - 1) * self.top_pct / 100.0).astype(int)
        top, cx, cy = h[pick], ix[pick], iy[pick]
        layer = np.full((self.ny, self.nx), np.nan)
        layer[cy, cx] = top
        if col is not None:                                         # color: the last frame that saw the cell
            self.C[cy, cx] = col[pick]
        self.layers.append(layer)
        self._H = None
        self.seen[cy, cx] += 1
        return int(len(pick))

    @property
    def H(self) -> np.ndarray:
        """Cell heights fused over the frames: the LOWER median of what the frames that saw the cell measured. The
        max let one oblique view win: a cell just past a brick's edge sees the brick's side wall from there, and every
        brick came out ~2.5 mm too big; with two frames the lower one (desk beats wall)."""
        if self._H is None:
            if not self.layers:
                self._H = np.full((self.ny, self.nx), np.nan)
            else:
                st = np.sort(np.stack(self.layers), axis=0)          # nan sorts last
                n = np.sum(~np.isnan(st), axis=0)
                k = np.clip((n - 1) // 2, 0, None)
                self._H = np.where(n > 0, np.take_along_axis(st, k[None], axis=0)[0], np.nan)
        return self._H

    @property
    def coverage(self) -> float:
        return float((self.seen > 0).mean())

    def objects(self, min_height: float = 0.004, min_area_m2: float = 2e-5, close_px: int = 1,
                top_frac: float = 0.75) -> list[MapObject]:
        """The blobs standing above the desk (min_height), at least min_area_m2 big. Outline, center and yaw come from
        the blob's TOP cells (>= top_frac of its height): seen from the side (one view, the real wrist camera 7 cm off
        to the side), a brick's wall points spread its outline toward the camera -- the real 32 x 16 mm 2x4 read
        40 x 20 mm. Set top_frac 0 for the whole blob."""
        m = (np.nan_to_num(self.H, nan=0.0) > min_height).astype(np.uint8)
        if close_px:
            k = np.ones((2 * close_px + 1, 2 * close_px + 1), np.uint8)
            m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] * self.res ** 2 < min_area_m2:
                continue
            iy, ix = np.nonzero(lab == i)
            hb = self.H[iy, ix]
            if not np.isfinite(hb).any():
                continue
            if top_frac > 0:
                top = np.isfinite(hb) & (hb >= top_frac * np.nanpercentile(hb, 90))
                if top.sum() >= 4:
                    iy, ix = iy[top], ix[top]
            x, y = self.center_of(ix, iy)
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.column_stack([x, y]).astype(np.float32))
            w, h = w + self.res, h + self.res                          # cell centers -> cell edges
            yaw = np.deg2rad(ang if w >= h else ang + 90.0)
            yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)
            # NaN-aware: the closing above fills holes INTO the blob (a shiny top -- the real white brick -- leaves
            # cells without depth); their missing heights made the percentile NaN and the object unclassifiable
            heights = self.H[iy, ix]
            col = np.median(self.C[iy, ix].reshape(-1, 3), axis=0)
            out.append(MapObject((float(cx), float(cy)), (float(max(w, h)), float(min(w, h))), yaw,
                                 float(np.nanpercentile(heights, 90)), tuple(int(v) for v in col), int(len(ix))))
        return out

    def render(self, scale: int = 2, objects=None) -> np.ndarray:
        """Top view as an image (robot base frame: x up, y left, like the workcell map): colors where seen, the height
        as shading, unseen cells dark; objects outlined with their match."""
        seen = self.seen > 0
        img = np.full((self.ny, self.nx, 3), 40, np.uint8)
        img[seen] = self.C[seen]
        h = np.nan_to_num(self.H, nan=0.0)
        shade = np.clip(0.75 + 4.0 * h, 0.75, 1.3)[..., None]
        img = np.clip(img * shade, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img.transpose(1, 0, 2)[::-1, ::-1])     # row = -x (x up), column = -y (y left)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for o in objects or []:
            ix, iy = self.cell_of(np.array(o.center_xy))
            u, v = (self.ny - 1 - iy) * scale, (self.nx - 1 - ix) * scale
            L, W = o.footprint[0] / self.res * scale, o.footprint[1] / self.res * scale
            # the long side's direction in the image: base (cos, sin) -> (column, row) = (-sin, -cos)
            ang = float(np.degrees(np.arctan2(-np.cos(o.yaw), -np.sin(o.yaw))))
            box = cv2.boxPoints(((float(u), float(v)), (float(L), float(W)), float(ang))).astype(int)
            cv2.polylines(img, [box], True, (60, 220, 60), 1)
            label = o.match.get("object", "?") if o.match else "?"
            cv2.putText(img, label, (int(u) - 20, int(v) - int(max(L, W) / 2) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1)
        return img


def fit_desk_plane(points: np.ndarray, z_hint: float, window: float = 0.04, tol: float = 0.003, iters: int = 300,
                   rng=None) -> tuple[np.ndarray, float, float] | None:
    """The desk as the dominant plane of base-frame points near z_hint (RANSAC, then least squares on the inliers).
    -> (unit normal pointing up, d with n.p + d = 0, inlier fraction) or None.

    Why: before the wrist camera is calibrated, its points come out shifted and tilted (2 cm low and ~5 cm of tilt
    across one view on the real SO-101). Heights measured from the FITTED desk stay right; x/y keep the error."""
    p = np.asarray(points, float)
    p = p[np.abs(p[:, 2] - z_hint) < window]
    if len(p) < 200:
        return None
    rng = rng or np.random.default_rng(0)
    sub = p[rng.choice(len(p), min(len(p), 6000), replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[2]) < 0.8:                                   # not desk-like (a wall, a box side)
            continue
        inl = np.abs((sub - a) @ n) < tol
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None:
        return None
    q = sub[best]
    c = q.mean(axis=0)
    n = np.linalg.svd(q - c)[2][-1]                         # least-squares normal of the inliers
    n = n if n[2] > 0 else -n
    return n, float(-n @ c), best_n / len(sub)


def level_to_plane(points: np.ndarray, n: np.ndarray, d: float, table_z: float) -> np.ndarray:
    """Points with z replaced by table_z + their height above the plane (n.p + d = 0): the desk becomes flat at
    table_z, objects keep their true height above it."""
    p = np.array(points, float)
    p[:, 2] = table_z + p @ n + d
    return p


def classify(obj: MapObject, catalog: dict, names, rest_poses_fn, tol_m: float = 0.006):
    """Which catalog object (of `names`) and resting face fits this blob's footprint + height best? -> match dict or {}.
    Error = the worst of the three size differences; accepted within tol_m."""
    best = {}
    for n in names:
        o = catalog[n]
        for r in rest_poses_fn(o).values():
            fp = sorted(r.footprint_m, reverse=True)
            err = max(abs(obj.footprint[0] - fp[0]), abs(obj.footprint[1] - fp[1]), abs(obj.height - 2 * r.height_m))
            # faces that look the same from above (standing / upside down: only the studs differ): standing
            # unless something says otherwise
            key = err + (0.0 if r.name == "upright" else 0.0005)
            if err <= tol_m and (not best or key < best["_key"]):
                best = {"object": n, "rest": r.name, "error": round(float(err), 4), "_key": key}
    best.pop("_key", None)
    return best
