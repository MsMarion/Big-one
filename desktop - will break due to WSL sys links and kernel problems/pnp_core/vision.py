"""The vision system as STAGES with swappable backends (docs/learning/24_vision_plan.md). Pure Python, no ROS.

    proposals  "where are objects"      -> Proposal  (pixel mask and/or base-frame blob, a source, a score)
    identity   "which object is this"   -> Identity  (catalog name, score, source)
    pose       "where exactly, how"     -> T_base_obj (4x4), from the proposal + identity

A Pipeline runs the backends named in config/vision.yaml for each stage. The result: one ObjectEstimate per
proposal that some identity backend recognised, carrying who said what. Every backend is judged on the same real test
set (tools/eval_vision.py, later), so swapping one is a config change, not a rewrite.

Backends here (more plug into the same interfaces):
    proposals:  depth_blobs   the flat-map pipeline on one RGB-D frame (desk plane fit, blobs above it) -- no training
                yolo_seg      an ultralytics segmentation model (closed set; its class rides along as a pre-identity)
    identity:   catalog_size  footprint + height vs config/objects.yaml (pnp_core/height_map.classify) -- no training
                yolo_class    the class the yolo_seg proposal came with (mapped through objects.yaml yolo_class)
    pose:       box_geometry  lying-flat box from the blob's center, yaw and the catalog's height
                dino_templates DINOv2/v3 embedding of the crop vs a reference library (renders / photos / saved crops)
Planned (same interfaces): sam2 proposals, owlv2 (text / image prompt) proposals, foundationpose pose (the PC's existing
TensorRT pipeline).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import cv2

from .height_map import HeightMap, classify, fit_desk_plane, level_to_plane
from .object_grasp import upright_pose
from .reorient import rest_poses


# --- data -------------------------------------------------------------------------------------------------------
@dataclass
class Frame:
    """One RGB-D view. depth in metres (0 = no data), K 3x3, T_base_cam = base <- camera OPTICAL frame."""
    rgb: np.ndarray | None
    depth: np.ndarray
    K: np.ndarray
    T_base_cam: np.ndarray
    stamp: float = 0.0
    camera: str = ""

    def points_base(self, stride: int = 1, z_range=(0.06, 1.2)):
        """-> (N x 3 points in the base frame, N x 2 pixel coordinates (u, v))."""
        vs, us = np.mgrid[0:self.depth.shape[0]:stride, 0:self.depth.shape[1]:stride]
        z = self.depth[::stride, ::stride]
        ok = (z > z_range[0]) & (z < z_range[1])
        u, v, z = us[ok], vs[ok], z[ok]
        K = self.K
        pc = np.column_stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        return pc @ self.T_base_cam[:3, :3].T + self.T_base_cam[:3, 3], np.column_stack([u, v])


@dataclass
class Proposal:
    source: str
    score: float = 1.0
    mask: np.ndarray | None = None          # H x W bool in the frame's image (None: no pixel mask)
    center_xy: tuple | None = None          # base frame, on the desk (blob proposals)
    footprint: tuple | None = None          # (long, short) m
    yaw: float | None = None                # rad, of the long side
    height: float | None = None             # m above the desk
    extra: dict = field(default_factory=dict)


@dataclass
class Identity:
    name: str
    score: float
    source: str
    extra: dict = field(default_factory=dict)


@dataclass
class ObjectEstimate:
    proposal: Proposal
    identity: Identity
    T_base_obj: np.ndarray | None = None
    pose_source: str = ""
    notes: list = field(default_factory=list)


# --- proposals ----------------------------------------------------------------------------------------------------
class DepthBlobs:
    """Blobs standing above the desk in ONE frame: fit the desk plane (the camera's absolute height may be off by a
    cm), measure heights from it, the flat map's blob extraction. Each blob also gets a pixel mask (its points)."""
    name = "depth_blobs"

    def __init__(self, region, table_z: float, res: float = 0.002, stride: int = 1, max_height: float = 0.12,
                 robot_keepout_m: float = 0.07, margin_m: float = 0.03, mask_pad_m: float = 0.002,
                 mask_min_height_m: float = 0.0015):
        self.region, self.table_z, self.res, self.stride, self.max_height = region, table_z, res, stride, max_height
        # the pixel mask: the footprint comes from the TOP cells, the walls stand just outside it and go down to the
        # desk -- pad the footprint and keep low points, or the mask misses the walls (sim: 16 % of the brick's pixels)
        self.mask_pad, self.mask_min_h = mask_pad_m, mask_min_height_m
        # the robot's own base pokes into the work area: on the sim test set its two front features came out as a
        # "1x2 brick" in EVERY frame. Blobs centered within this radius of the base origin are dropped (a brick there
        # is inside the arm's minimum reach anyway). A real self-filter (the URDF's meshes) would also cover the arm.
        self.robot_keepout_m = robot_keepout_m
        # map a margin AROUND the region, report only blobs centered inside it: on the sim test set 51 of 95 errors
        # were bricks cut by the region's edge (measured too small -> the wrong type, or missed)
        self.margin = margin_m

    def propose(self, frame: Frame) -> list:
        P, uv = frame.points_base(self.stride)
        fit = fit_desk_plane(P, self.table_z)
        if fit is not None:
            P = level_to_plane(P, fit[0], fit[1], self.table_z)
        keep = P[:, 2] < self.table_z + self.max_height
        P, uv = P[keep], uv[keep]
        r, mg = self.region, self.margin
        hm = HeightMap((r[0] - mg, r[1] + mg), (r[2] - mg, r[3] + mg), self.res, self.table_z)
        hm.add(P)
        out = []
        ix, iy = hm.cell_of(P[:, :2])
        inside = (ix >= 0) & (ix < hm.nx) & (iy >= 0) & (iy < hm.ny)
        for o in hm.objects():
            if np.hypot(*o.center_xy) < self.robot_keepout_m:
                continue
            if not (r[0] <= o.center_xy[0] <= r[1] and r[2] <= o.center_xy[1] <= r[3]):
                continue
            # pixels of the points that fall in the blob's footprint and stand above the desk
            d = P[:, :2] - np.asarray(o.center_xy)
            c, s = np.cos(o.yaw), np.sin(o.yaw)
            along, across = d @ [c, s], d @ [-s, c]
            sel = inside & (np.abs(along) <= o.footprint[0] / 2 + self.mask_pad) & \
                (np.abs(across) <= o.footprint[1] / 2 + self.mask_pad) & (P[:, 2] > self.table_z + self.mask_min_h)
            mask = np.zeros(frame.depth.shape, np.uint8)
            mask[uv[sel, 1], uv[sel, 0]] = 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)   # depth holes
            out.append(Proposal(self.name, 1.0, mask, tuple(o.center_xy), tuple(o.footprint), float(o.yaw),
                                float(o.height), {"cells": o.cells, "blob": o,
                                                  "desk_fit": None if fit is None else (fit[0].tolist(), fit[1])}))
        return out


class YoloSeg:
    """An ultralytics segmentation model; the class it predicted rides along (extra['class'])."""
    name = "yolo_seg"

    def __init__(self, model_path: str, conf: float = 0.25, device=None):
        from ultralytics import YOLO          # optional dependency: only when this backend is configured
        self.model, self.conf, self.device = YOLO(model_path), conf, device

    def propose(self, frame: Frame) -> list:
        if frame.rgb is None:
            return []
        r = self.model.predict(frame.rgb, conf=self.conf, device=self.device, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return []
        out = []
        for m, score, cls in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            out.append(Proposal(self.name, float(score), m > 0.5, extra={"class": self.model.names[int(cls)]}))
        return out


# --- identity -----------------------------------------------------------------------------------------------------
class CatalogSize:
    """Footprint + height vs the catalog (blob proposals only)."""
    name = "catalog_size"

    def __init__(self, catalog: dict, names=None, tol_m: float = 0.006):
        self.catalog = catalog
        self.names = list(names) if names else [k for k, v in catalog.items() if "brick" in v]
        self.tol = tol_m

    def identify(self, frame: Frame, p: Proposal) -> Identity | None:
        blob = p.extra.get("blob")
        if blob is None:
            return None
        m = classify(blob, self.catalog, self.names, rest_poses, self.tol)
        if not m:
            return None
        return Identity(m["object"], float(max(0.0, 1.0 - m["error"] / self.tol)), self.name, {"rest": m["rest"]})


class YoloClass:
    """The class a yolo_seg proposal came with, mapped to the catalog name (objects.yaml yolo_class)."""
    name = "yolo_class"

    def __init__(self, catalog: dict):
        self.by_class = {v.get("yolo_class"): k for k, v in catalog.items() if v.get("yolo_class")}

    def identify(self, frame: Frame, p: Proposal) -> Identity | None:
        c = p.extra.get("class")
        return Identity(self.by_class[c], p.score, self.name) if c in self.by_class else None


def crop_for_embedding(rgb: np.ndarray, mask: np.ndarray, margin: float = 0.15, bg: int = 124) -> np.ndarray | None:
    """The proposal's pixels as a square crop, the background flattened to gray (what a template looks like too)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * margin)) + 1
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    out = np.full((side, side, 3), bg, np.uint8)
    sx0, sy0 = cx - side // 2, cy - side // 2
    for dy in range(side):                                 # copy the masked pixels that fall inside the image
        y = sy0 + dy
        if 0 <= y < rgb.shape[0]:
            xs_ = np.arange(max(0, sx0), min(rgb.shape[1], sx0 + side))
            keep = mask[y, xs_]
            out[dy, xs_[keep] - sx0] = rgb[y, xs_[keep]]
    return out


class DinoTemplates:
    """Identity by appearance, NO training per object: DINOv2/v3 embeddings of the proposal's crop vs a reference
    library (tools/build_templates.py: mesh renders, photos, or crops the robot saved). Cosine similarity; the best
    object wins if it beats `min_score` and the runner-up by `margin`."""
    name = "dino_templates"

    def __init__(self, library: str, model_dir: str = "models/vision/dinov2-small", device=None,
                 min_score: float = 0.5, margin: float = 0.03):
        import torch
        from transformers import AutoImageProcessor, AutoModel       # optional dependency
        lib = np.load(library, allow_pickle=True)
        self.names, self.E = list(lib["names"]), lib["embeddings"].astype(np.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device).eval()
        self.min_score, self.margin, self.torch = min_score, margin, torch

    def embed(self, crops: list) -> np.ndarray:
        with self.torch.no_grad():
            x = self.proc(images=crops, return_tensors="pt").to(self.device)
            e = self.model(**x).last_hidden_state[:, 0].float().cpu().numpy()          # the CLS token
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def identify(self, frame: Frame, p: Proposal) -> Identity | None:
        if frame.rgb is None or p.mask is None:
            return None
        crop = crop_for_embedding(frame.rgb, p.mask)
        if crop is None:
            return None
        sims = self.E @ self.embed([crop])[0]
        best = {}
        for n, s in zip(self.names, sims):
            best[n] = max(best.get(n, -1.0), float(s))
        ranked = sorted(best.items(), key=lambda kv: -kv[1])
        top, second = ranked[0], (ranked[1] if len(ranked) > 1 else (None, -1.0))
        if top[1] < self.min_score or top[1] - second[1] < self.margin:
            return None
        return Identity(top[0], top[1], self.name, {"runner_up": second[0], "runner_up_score": second[1]})


def kind_of(name: str) -> str:
    """The KIND of a catalog object: every LEGO brick is one kind ("brick"); other objects are their own kind."""
    return "brick" if name.startswith("lego_") else name


class KindFusion:
    """Geometry names WHICH object (millimetres: catalog_size), DINO checks WHAT KIND it is (appearance:
    dino_templates). Measured on the sim test sets: DINO tells kinds apart (93-100 %) but not brick sizes (the crop drops
    the scale); the depth layer is the opposite.
      - geometry matched, DINO agrees on the kind (or isn't sure): the geometry's object; score raised when it agrees
      - geometry matched, DINO is SURE (>= reject_sim) it's another kind: rejected (a flat print, a box, ...)
      - no geometry match, DINO sure of a non-brick object whose catalog size fits within `size_tol`: DINO's object"""
    name = "kind_fusion"

    def __init__(self, catalog: dict, dino: dict, reject_sim: float = 0.6, size_tol: float = 0.015, names=None):
        self.geo = CatalogSize(catalog, names)
        self.dino = DinoTemplates(**dino)
        self.catalog, self.reject_sim, self.size_tol = catalog, reject_sim, size_tol

    def identify(self, frame: Frame, p: Proposal) -> Identity | None:
        g = self.geo.identify(frame, p)
        d = self._dino(frame, p)
        if g is not None:
            if d is None:
                return Identity(g.name, g.score * 0.8, self.name, {**g.extra, "dino": None})
            if kind_of(d[0]) == kind_of(g.name):
                return Identity(g.name, min(1.0, 0.5 * g.score + 0.5 * d[1] + 0.2), self.name,
                                {**g.extra, "dino": d})
            if d[1] >= self.reject_sim:
                return None                                     # DINO is sure it's something else
            return Identity(g.name, g.score * 0.6, self.name, {**g.extra, "dino": d, "kind_disagrees": True})
        if d is not None and d[1] >= self.reject_sim and not d[0].startswith("lego_") and p.footprint is not None:
            dims = sorted(self.catalog[d[0]]["dims_m"], reverse=True)
            meas = sorted([p.footprint[0], p.footprint[1], p.height or 0.0], reverse=True)
            if max(abs(a - b) for a, b in zip(dims, meas)) <= self.size_tol:
                return Identity(d[0], d[1], self.name, {"dino": d, "rest": "upright"})
        return None

    def _dino(self, frame, p):
        """-> (best object, similarity) regardless of the dino backend's own thresholds, or None."""
        if frame.rgb is None or p.mask is None:
            return None
        crop = crop_for_embedding(frame.rgb, p.mask)
        if crop is None:
            return None
        sims = self.dino.E @ self.dino.embed([crop])[0]
        k = int(np.argmax(sims))
        return self.dino.names[k], float(sims[k])


# --- pose ---------------------------------------------------------------------------------------------------------
class BoxGeometry:
    """A box lying on a face: center + yaw from the blob, the height from the catalog (a better z than the blob's)."""
    name = "box_geometry"

    def __init__(self, catalog: dict, table_z: float):
        self.catalog, self.table_z = catalog, table_z

    def estimate(self, frame: Frame, p: Proposal, ident: Identity) -> np.ndarray | None:
        if p.center_xy is None or ident.extra.get("rest", "upright") not in ("upright", "upside_down"):
            return None
        oc = self.catalog[ident.name]
        return upright_pose(oc, p.center_xy, self.table_z + float(oc["dims_m"][2]) / 2, p.yaw or 0.0)


# --- pipeline -------------------------------------------------------------------------------------------------------
class Pipeline:
    """proposals from every proposal backend; for each, the FIRST identity backend that recognises it (in config
    order), then the first pose backend that gives a pose. Disagreements are noted, not hidden."""

    def __init__(self, proposals: list, identity: list, pose: list):
        self.proposals, self.identity, self.pose = proposals, identity, pose

    def run(self, frame: Frame) -> list:
        out = []
        for pb in self.proposals:
            for p in pb.propose(frame):
                ids = [i for i in (ib.identify(frame, p) for ib in self.identity) if i is not None]
                if not ids:
                    continue
                est = ObjectEstimate(p, ids[0])
                if len({i.name for i in ids}) > 1:
                    est.notes.append("identity disagreement: " + ", ".join(f"{i.source}={i.name}" for i in ids))
                for pob in self.pose:
                    T = pob.estimate(frame, p, ids[0])
                    if T is not None:
                        est.T_base_obj, est.pose_source = T, pob.name
                        break
                out.append(est)
        return out


def build_pipeline(cfg: dict, catalog: dict, table_z: float, region) -> Pipeline:
    """cfg = config/vision.yaml's profile: {proposals: [{name, ...params}], identity: [...], pose: [...]}."""
    def make(stage, spec):
        spec = dict(spec)
        kind = spec.pop("name")
        if stage == "proposals" and kind == "depth_blobs":
            return DepthBlobs(region, table_z, **spec)
        if stage == "proposals" and kind == "yolo_seg":
            return YoloSeg(**spec)
        if stage == "identity" and kind == "catalog_size":
            return CatalogSize(catalog, **spec)
        if stage == "identity" and kind == "yolo_class":
            return YoloClass(catalog)
        if stage == "identity" and kind == "dino_templates":
            return DinoTemplates(**spec)
        if stage == "identity" and kind == "kind_fusion":
            return KindFusion(catalog, **spec)
        if stage == "pose" and kind == "box_geometry":
            return BoxGeometry(catalog, table_z)
        raise ValueError(f"no {stage} backend '{kind}' (yet): see pnp_core/vision.py")
    return Pipeline(*[[make(st, s) for s in cfg.get(st, [])] for st in ("proposals", "identity", "pose")])
