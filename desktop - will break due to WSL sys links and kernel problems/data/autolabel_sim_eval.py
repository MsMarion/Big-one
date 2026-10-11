"""How good are the auto-labels (tools/autolabel.py: depth pipeline -> polygons)? Measured against the exact instance
masks of the SIM brick test set (data/raw/lego_test).
    python3 data/autolabel_sim_eval.py [--raw data/raw/lego_test] [--frames 300] [--profile geometry]   (cwd = repo)
A label is matched to the ground-truth brick its polygon overlaps most (IoU >= 0.5), and is
  right   = matched, same type        wrong = matched, other type        ghost = no brick under it
Recall is over the bricks a label COULD be made for: >= 300 px in the image (whole or nearly whole); the rest are
listed separately (an auto-labeller leaving out a half-hidden brick is fine, a wrong label is not).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from autolabel import CLASSES, label_frame  # noqa: E402
from pnp_core.catalog import load_yaml  # noqa: E402
from pnp_core.vision import build_pipeline  # noqa: E402
from sim_raw import SimRaw  # noqa: E402

MIN_PX = 300


def raster(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [(np.array(poly).reshape(-1, 2) * [shape[1], shape[0]]).round().astype(np.int32)], 1)
    return m.astype(bool)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/lego_test")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--profile", default="geometry")
    a = ap.parse_args()
    src = SimRaw(REPO / a.raw)
    pipe = build_pipeline(load_yaml("config/vision.yaml")["profiles"][a.profile], load_yaml("config/objects.yaml"),
                          src.table_z, src.region)
    right = wrong = ghost = 0
    gt_big = gt_small = found_big = found_small = 0
    ious, per = [], {c: dict(gt=0, found=0, labels=0, right=0) for c in CLASSES}
    confusion = {}
    n = min(a.frames, len(src))
    for i in range(n):
        f = src.frame(i)
        gt = [(c, m) for c, m in src.truth(i) if c in CLASSES]
        labs = label_frame(pipe, f)
        hit = set()
        for c, poly, _ in labs:
            pm = raster(poly, f.depth.shape)
            per[CLASSES[c]]["labels"] += 1
            best = max(((iou(pm, m), j) for j, (_, m) in enumerate(gt)), default=(0.0, -1))
            if best[0] < 0.5:
                ghost += 1
                continue
            g = gt[best[1]][0]
            hit.add(best[1])
            ious.append(best[0])
            if g == CLASSES[c]:
                right += 1
                per[g]["right"] += 1
            else:
                wrong += 1
                confusion[(g, CLASSES[c])] = confusion.get((g, CLASSES[c]), 0) + 1
        for j, (g, m) in enumerate(gt):
            big = m.sum() >= MIN_PX
            gt_big += big
            gt_small += not big
            ok = j in hit
            found_big += ok and big
            found_small += ok and not big
            if big:
                per[g]["gt"] += 1
                per[g]["found"] += ok
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    labels = right + wrong + ghost
    print(f"\n{n} frames ({a.raw}, profile {a.profile}): {labels} labels written")
    print(f"  precision (right type): {right / max(labels, 1):.1%}   wrong type {wrong}, ghost {ghost}")
    print(f"  recall: {found_big}/{gt_big} bricks >= {MIN_PX} px ({found_big / max(gt_big, 1):.1%}); "
          f"{found_small}/{gt_small} smaller (half hidden / far)")
    print(f"  mask IoU of matched labels: mean {np.mean(ious):.3f}, 10th pct {np.percentile(ious, 10):.3f}")
    for c, s in per.items():
        if s["gt"] or s["labels"]:
            print(f"    {c:9s} recall {s['found']}/{s['gt']}  precision {s['right']}/{s['labels']}")
    for (g, p), k in sorted(confusion.items(), key=lambda kv: -kv[1]):
        print(f"    {k:3d} x  {g} labelled {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
