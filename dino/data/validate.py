"""Check a dataset before training and summarize it (counts, class balance, sizes, depth/K presence).

    python -m dino.data.validate --root $DINO_DATA/mycam --tasks cls seg
"""
import argparse
import csv
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

from .datasets import IMG_EXT, list_frames, read_classes

MAX_SCAN = 3000  # frames per split to scan masks for (counts are exact, mask stats are sampled beyond this)


def _split_summary(root, split, classes, tasks):
    d = os.path.join(root, split)
    s = {"split": split, "errors": [], "warnings": []}
    if not os.path.isdir(d):
        s["errors"].append(f"missing split folder {d}")
        return s
    frames_layout = os.path.isdir(os.path.join(d, "rgb"))
    s["layout"] = "frames" if frames_layout else "imagefolder"
    if frames_layout:
        frames = list_frames(d)
    else:
        frames = [(f"{c}/{os.path.splitext(f)[0]}", os.path.join(d, c, f)) for c in classes if os.path.isdir(os.path.join(d, c))
                  for f in sorted(os.listdir(os.path.join(d, c))) if f.lower().endswith(IMG_EXT)]
    s["n"] = len(frames)
    if not frames:
        s["errors"].append("no images")
        return s
    sizes = Counter(Image.open(p).size for _, p in frames[:50])
    s["image_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.most_common(3)}
    s["has_depth"] = frames_layout and os.path.isdir(os.path.join(d, "depth"))
    s["has_K"] = os.path.exists(os.path.join(d, "cam_K.txt"))

    if "cls" in tasks:
        if frames_layout:
            path = os.path.join(d, "labels.csv")
            if not os.path.exists(path):
                s["errors"].append("cls: labels.csv missing")
            else:
                with open(path) as f:
                    rows = list(csv.DictReader(f))
                ids = {fid for fid, _ in frames}
                unknown = sorted({r["class"] for r in rows} - set(classes))
                missing = [r["id"] for r in rows if r["id"] not in ids]
                if unknown:
                    s["errors"].append(f"cls: classes not in classes.txt: {unknown[:10]}")
                if missing:
                    s["errors"].append(f"cls: {len(missing)} labels without an image, e.g. {missing[:3]}")
                s["cls_counts"] = dict(Counter(r["class"] for r in rows))
        else:
            s["cls_counts"] = dict(Counter(fid.split("/")[0] for fid, _ in frames))
        empty = [c for c in classes if not s.get("cls_counts", {}).get(c)]
        if empty:
            s["warnings"].append(f"cls: classes with no examples here: {empty[:10]}")

    if "seg" in tasks:
        if not frames_layout:
            s["errors"].append("seg: needs the frames layout with masks/")
        else:
            n_cls = len(classes)
            present = Counter()
            missing, bad_size, bad_vals, fg = 0, 0, set(), []
            for fid, p in frames[:MAX_SCAN]:
                mp = os.path.join(d, "masks", fid + ".png")
                if not os.path.exists(mp):
                    missing += 1
                    continue
                m = np.asarray(Image.open(mp))
                if m.ndim != 2:
                    s["errors"].append(f"seg: mask {fid} is not single-channel ({m.shape})")
                    break
                if m.shape[::-1] != Image.open(p).size:
                    bad_size += 1
                vals = np.unique(m)
                bad_vals.update(int(v) for v in vals if v > n_cls and v != 255)
                present.update(int(v) for v in vals if 0 < v <= n_cls)
                fg.append(float(((m > 0) & (m != 255)).mean()))
            if missing:
                s["errors"].append(f"seg: {missing} frames without masks/<id>.png")
            if bad_size:
                s["errors"].append(f"seg: {bad_size} masks differ in size from their rgb")
            if bad_vals:
                s["errors"].append(f"seg: mask values outside 0..{n_cls} (and 255): {sorted(bad_vals)[:10]}")
            s["seg_frames_with_class"] = {classes[k - 1]: v for k, v in sorted(present.items())}
            s["seg_mean_fg_frac"] = float(np.mean(fg)) if fg else 0.0
    return s


def summarize(root, splits, tasks):
    out = {"root": root, "errors": [], "warnings": []}
    try:
        out["classes"] = read_classes(root, splits[0])
    except OSError as e:
        out["errors"].append(f"cannot read classes: {e}")
        return out
    if not out["classes"]:
        out["errors"].append("no classes (add classes.txt or ImageFolder class folders)")
    out["splits"] = [_split_summary(root, sp, out["classes"], tasks) for sp in dict.fromkeys(splits)]
    for sp in out["splits"]:
        out["errors"] += [f"[{sp['split']}] {e}" for e in sp["errors"]]
        out["warnings"] += [f"[{sp['split']}] {w}" for w in sp["warnings"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--tasks", nargs="+", default=["cls", "seg"])
    a = ap.parse_args()
    s = summarize(a.root, a.splits, a.tasks)
    print(json.dumps(s, indent=2))
    raise SystemExit(1 if s["errors"] else 0)


if __name__ == "__main__":
    main()
