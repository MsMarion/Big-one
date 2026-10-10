"""Offline synthetic RGB-D smoketest data in the standard frames layout.

Each frame: textured background plane + one primary shape (the classification label) and
optionally a smaller distractor of another class. Writes rgb, depth (uint16 mm), semantic
masks, labels.csv and cam_K.txt per split.

    python -m dino.data.synthetic --out $DINO_DATA/synthetic --n_train 240 --n_val 60
"""
import argparse
import csv
import math
import os

import numpy as np
from PIL import Image, ImageDraw

from .rgbd import write_K, write_depth_mm

CLASSES = ["circle", "square", "triangle", "star"]


def _polygon(cls, cx, cy, r, rot):
    if cls == "square":
        n, radii = 4, [r]
    elif cls == "triangle":
        n, radii = 3, [r]
    else:  # star
        n, radii = 10, [r, r * 0.45]
    return [(cx + radii[k % len(radii)] * math.cos(rot + 2 * math.pi * k / n),
             cy + radii[k % len(radii)] * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def _shape_mask(cls, W, H, cx, cy, r, rot):
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    if cls == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    else:
        d.polygon(_polygon(cls, cx, cy, r, rot), fill=255)
    return np.asarray(layer) > 0


def make_frame(rng, W, H):
    coarse = rng.random((H // 16 + 1, W // 16 + 1, 3)) * 255
    bg = np.asarray(Image.fromarray(coarse.astype(np.uint8)).resize((W, H), Image.BICUBIC), dtype=np.float32)
    rgb = np.clip(bg * 0.7 + rng.normal(0, 12, (H, W, 3)), 0, 255)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = 1.2 + 0.4 * yy / H + rng.normal(0, 0.003, (H, W))  # tilted table/floor plane, metres
    sem = np.zeros((H, W), np.uint8)

    primary = int(rng.integers(len(CLASSES)))
    objs = []
    if rng.random() < 0.5:
        objs.append((int(rng.choice([c for c in range(len(CLASSES)) if c != primary])), rng.uniform(14, 26)))
    objs.append((primary, rng.uniform(34, 62)))  # primary drawn last so it is never occluded

    for c, r in objs:
        cx, cy = rng.uniform(r, W - r), rng.uniform(r, H - r)
        m = _shape_mask(CLASSES[c], W, H, cx, cy, r, rng.uniform(0, 2 * math.pi))
        color = rng.uniform(30, 230, 3)
        shade = 1.0 - 0.35 * np.clip(((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r), 0, 1)
        rgb[m] = (color[None] * shade[m][:, None]).clip(0, 255)
        z = rng.uniform(0.55, 0.95)
        bump = 0.05 * np.sqrt(np.clip(1 - ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r), 0, 1)) if c == 0 else 0
        depth[m] = (z - bump)[m] if c == 0 else z
        sem[m] = c + 1
    return rgb.astype(np.uint8), depth, sem, primary


def generate(out, n_train, n_val, W=320, H=240, seed=0, flat=False):
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES) + "\n")
    K = np.array([[0.9 * W, 0, W / 2], [0, 0.9 * W, H / 2], [0, 0, 1]])
    # flat: one unsplit "all/" folder, like a fresh camera dump (split later by the experiment)
    for split, n in ((("all", n_train + n_val),) if flat else (("train", n_train), ("val", n_val))):
        d = os.path.join(out, split)
        for sub in ("rgb", "depth", "masks"):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        rows = []
        for i in range(n):
            fid = f"{i:06d}"
            rgb, depth, sem, label = make_frame(rng, W, H)
            Image.fromarray(rgb).save(os.path.join(d, "rgb", fid + ".png"))
            write_depth_mm(os.path.join(d, "depth", fid + ".png"), depth)
            Image.fromarray(sem).save(os.path.join(d, "masks", fid + ".png"))
            rows.append((fid, CLASSES[label]))
        with open(os.path.join(d, "labels.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "class"])
            w.writerows(rows)
        write_K(os.path.join(d, "cam_K.txt"), K)
    print(f"synthetic: {n_train + n_val} frames ({'all/' if flat else 'train/val'}), {len(CLASSES)} classes -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("DINO_DATA", "data"), "synthetic"))
    ap.add_argument("--n_train", type=int, default=240)
    ap.add_argument("--n_val", type=int, default=60)
    ap.add_argument("--size", type=int, nargs=2, default=(320, 240), metavar=("W", "H"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flat", action="store_true", help="write a single unsplit all/ folder")
    a = ap.parse_args()
    generate(a.out, a.n_train, a.n_val, *a.size, a.seed, a.flat)


if __name__ == "__main__":
    main()
