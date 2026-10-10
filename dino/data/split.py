"""Split one folder of new data into train/val/test (symlinks, no copies).

Works on the frames layout (<root>/<src>/rgb/...) and on ImageFolder (<root>/<src>/<class>/...).
Classification labels are split per class (stratified) so rare classes land in every split.

    python -m dino.data.split --root $DINO_DATA/mycam --src all --ratios 0.7 0.15 0.15
"""
import argparse
import csv
import os
import random
import shutil
from collections import defaultdict

from .datasets import IMG_EXT, list_frames

SPLITS = ("train", "val", "test")


def _assign(keys_by_group, ratios, seed):
    rng = random.Random(seed)
    out = {s: [] for s in SPLITS}
    for _, keys in sorted(keys_by_group.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
        out["val"] += keys[:n_val]
        out["test"] += keys[n_val:n_val + n_test]
        out["train"] += keys[n_val + n_test:]
    return out


def _link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def split_frames(root, src, ratios, seed):
    sd = os.path.join(root, src)
    frames = dict(list_frames(sd))
    labels = {}
    if os.path.exists(os.path.join(sd, "labels.csv")):
        with open(os.path.join(sd, "labels.csv")) as f:
            labels = {r["id"]: r["class"] for r in csv.DictReader(f)}
    groups = defaultdict(list)
    for fid in frames:
        groups[labels.get(fid, "_")].append(fid)
    assign = _assign(groups, ratios, seed)
    for split, ids in assign.items():
        d = os.path.join(root, split)
        if os.path.exists(d):
            shutil.rmtree(d)
        for fid in ids:
            rgb = frames[fid]
            _link(rgb, os.path.join(d, "rgb", os.path.basename(rgb)))
            for sub in ("depth", "masks"):
                p = os.path.join(sd, sub, fid + ".png")
                if os.path.exists(p):
                    _link(p, os.path.join(d, sub, fid + ".png"))
        if labels:
            with open(os.path.join(d, "labels.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "class"])
                w.writerows((fid, labels[fid]) for fid in sorted(ids))
        if os.path.exists(os.path.join(sd, "cam_K.txt")):
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(os.path.join(sd, "cam_K.txt"), os.path.join(d, "cam_K.txt"))
    return {s: len(v) for s, v in assign.items()}


def split_imagefolder(root, src, ratios, seed):
    sd = os.path.join(root, src)
    groups = {c: [os.path.join(c, f) for f in os.listdir(os.path.join(sd, c)) if f.lower().endswith(IMG_EXT)]
              for c in os.listdir(sd) if os.path.isdir(os.path.join(sd, c))}
    assign = _assign(groups, ratios, seed)
    for split, rels in assign.items():
        d = os.path.join(root, split)
        if os.path.exists(d):
            shutil.rmtree(d)
        for rel in rels:
            _link(os.path.join(sd, rel), os.path.join(d, rel))
    if not os.path.exists(os.path.join(root, "classes.txt")):
        with open(os.path.join(root, "classes.txt"), "w") as f:
            f.write("\n".join(sorted(groups)) + "\n")
    return {s: len(v) for s, v in assign.items()}


def make_splits(root, src="all", ratios=(0.7, 0.15, 0.15), seed=0):
    if os.path.isdir(os.path.join(root, src, "rgb")):
        return split_frames(root, src, ratios, seed)
    return split_imagefolder(root, src, ratios, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--src", default="all")
    ap.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(make_splits(a.root, a.src, a.ratios, a.seed))


if __name__ == "__main__":
    main()
