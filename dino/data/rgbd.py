"""RGB-D helpers: depth/intrinsics I/O, ingestion of raw camera captures, FoundationPose export.

FoundationPose per-scene layout (what `export_frame` writes):
    <scene>/rgb/<id>.png  <scene>/depth/<id>.png (uint16 mm)  <scene>/masks/<id>.png (0/255)  <scene>/cam_K.txt

Ingest raw captures into the standard frames layout:
    python -m dino.data.rgbd --rgb_dir cap/color --depth_dir cap/depth --K cap/K.txt \
        --depth_scale 0.001 --out $DINO_DATA/mycam/train [--resize 640 480]
--depth_scale converts raw depth units to metres (RealSense z16 = 0.001, Azure Kinect = 0.001).
--K accepts a 3x3 text file or a JSON with fx, fy, cx, cy (or "K": 3x3).
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

from .datasets import IMG_EXT


def read_K(path):
    if path.endswith(".json"):
        with open(path) as f:
            j = json.load(f)
        if "K" in j:
            return np.asarray(j["K"], dtype=np.float64).reshape(3, 3)
        return np.array([[j["fx"], 0, j["cx"]], [0, j["fy"], j["cy"]], [0, 0, 1]], dtype=np.float64)
    return np.loadtxt(path).reshape(3, 3)


def write_K(path, K):
    np.savetxt(path, np.asarray(K).reshape(3, 3), fmt="%.6f")


def scale_K(K, sx, sy):
    K = np.asarray(K, dtype=np.float64).copy()
    K[0] *= sx
    K[1] *= sy
    return K


def read_depth_m(path, scale=0.001):
    return np.asarray(Image.open(path), dtype=np.float32) * scale


def write_depth_mm(path, depth_m):
    Image.fromarray(np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)).save(path)


def export_frame(scene_dir, fid, rgb_path, depth_path, mask, K=None):
    """Write one FoundationPose frame. `mask` is a bool/uint8 HxW array at RGB resolution."""
    for sub in ("rgb", "depth", "masks"):
        os.makedirs(os.path.join(scene_dir, sub), exist_ok=True)
    Image.open(rgb_path).convert("RGB").save(os.path.join(scene_dir, "rgb", fid + ".png"))
    if depth_path and os.path.exists(depth_path):
        shutil.copyfile(depth_path, os.path.join(scene_dir, "depth", fid + ".png"))
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255).save(os.path.join(scene_dir, "masks", fid + ".png"))
    if K is not None and not os.path.exists(os.path.join(scene_dir, "cam_K.txt")):
        write_K(os.path.join(scene_dir, "cam_K.txt"), K)


def _list(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


def ingest(rgb_dir, depth_dir, K_path, out, depth_scale=0.001, resize=None):
    rgbs, depths = _list(rgb_dir), _list(depth_dir) if depth_dir else []
    dmap = {os.path.splitext(f)[0]: f for f in depths}
    pair_by_name = depths and all(os.path.splitext(f)[0] in dmap for f in rgbs)
    if depths and not pair_by_name and len(depths) != len(rgbs):
        raise SystemExit(f"cannot pair {len(rgbs)} rgb with {len(depths)} depth frames (names differ and counts differ)")
    for sub in ("rgb", "depth"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    K = read_K(K_path) if K_path else None
    for i, f in enumerate(rgbs):
        fid = f"{i:06d}"
        img = Image.open(os.path.join(rgb_dir, f)).convert("RGB")
        W0, H0 = img.size
        if resize:
            img = img.resize(tuple(resize), Image.BILINEAR)
        img.save(os.path.join(out, "rgb", fid + ".png"))
        if depths:
            df = dmap[os.path.splitext(f)[0]] if pair_by_name else depths[i]
            d = read_depth_m(os.path.join(depth_dir, df), depth_scale)
            if resize:
                d = np.asarray(Image.fromarray(d).resize(tuple(resize), Image.NEAREST))
            write_depth_mm(os.path.join(out, "depth", fid + ".png"), d)
    if K is not None:
        if resize:
            K = scale_K(K, resize[0] / W0, resize[1] / H0)
        write_K(os.path.join(out, "cam_K.txt"), K)
    print(f"ingested {len(rgbs)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--depth_dir")
    ap.add_argument("--K")
    ap.add_argument("--depth_scale", type=float, default=0.001)
    ap.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ingest(a.rgb_dir, a.depth_dir, a.K, a.out, a.depth_scale, a.resize)


if __name__ == "__main__":
    main()
