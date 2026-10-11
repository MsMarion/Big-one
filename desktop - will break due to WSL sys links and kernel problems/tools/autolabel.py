"""Auto-label RGB-D frames for detector training: the robot labels its own images (docs/learning/24_vision_plan.md V0).
Every object the depth pipeline identifies WHOLE (config/vision.yaml profile, default `geometry`: depth blobs + catalog
size) becomes a YOLO-seg polygon from its pixel mask; nothing else is labelled (an unsure blob is left out, not
guessed). Review the contact sheet before training.

    python3 tools/autolabel.py --sim-raw data/raw/lego_v1 --out data/datasets/lego_auto_v1 [--frames 2500] [--val 0.1]
    (later: --bag <rosbag of a real scan session>)
-> images/{train,val}, labels/{train,val}, dataset.yaml, manifest.json, contact_sheet.png

Sim frames (Isaac Replicator raw output with depth) let us MEASURE the labels against the exact masks first:
data/autolabel_sim_eval.py.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
from pnp_core.catalog import load_yaml  # noqa: E402
from pnp_core.vision import build_pipeline  # noqa: E402
from sim_raw import SimRaw  # noqa: E402

CLASSES = ["lego_2x4", "lego_2x2", "lego_1x2", "lego_2x3"]   # lego_seg's order + the new 2x3


def polygon(mask: np.ndarray, eps_px: float = 1.0):
    """The mask's largest outline as a normalized polygon [x1 y1 x2 y2 ...], or None."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), eps_px, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    h, w = mask.shape
    return (c / [w, h]).ravel().tolist()


def label_frame(pipe, f, classes=CLASSES):
    """-> [(class index, polygon, estimate)] for the whole, identified objects."""
    out = []
    for e in pipe.run(f):
        if e.identity.name not in classes or e.proposal.mask is None:
            continue
        poly = polygon(e.proposal.mask)
        if poly:
            out.append((classes.index(e.identity.name), poly, e))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=10 ** 9)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--profile", default="geometry")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    src = SimRaw(REPO / a.sim_raw)
    cat = load_yaml("config/objects.yaml")
    pipe = build_pipeline(load_yaml("config/vision.yaml")["profiles"][a.profile], cat, src.table_z, src.region)
    out = REPO / a.out
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    rng = np.random.default_rng(a.seed)
    n = min(a.frames, len(src))
    counts, empty, sheet = {c: 0 for c in CLASSES}, 0, []
    for i in range(n):
        f = src.frame(i)
        labs = label_frame(pipe, f)
        split = "val" if rng.random() < a.val else "train"
        name = f"{Path(a.sim_raw).name}_{i:05d}"
        cv2.imwrite(str(out / "images" / split / f"{name}.png"), f.rgb[..., ::-1])
        (out / "labels" / split / f"{name}.txt").write_text(
            "".join(f"{c} " + " ".join(f"{v:.5f}" for v in p) + "\n" for c, p, _ in labs))
        empty += not labs
        for c, _, _ in labs:
            counts[CLASSES[c]] += 1
        if len(sheet) < 12 and labs:
            vis = f.rgb[..., ::-1].copy()
            for c, p, _ in labs:
                pts = (np.array(p).reshape(-1, 2) * [vis.shape[1], vis.shape[0]]).astype(np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
                cv2.putText(vis, CLASSES[c].replace("lego_", ""), tuple(pts[0]), 0, 0.4, (0, 255, 0), 1)
            sheet.append(cv2.resize(vis, (320, 240)))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n} frames", flush=True)
    (out / "dataset.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n" +
                                      "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES)))
    (out / "manifest.json").write_text(json.dumps({"source": a.sim_raw, "frames": n, "profile": a.profile,
                                                   "labels": counts, "frames_without_labels": empty,
                                                   "classes": CLASSES}, indent=1))
    if sheet:
        rows = [np.hstack(sheet[k:k + 4] + [np.zeros_like(sheet[0])] * (4 - len(sheet[k:k + 4])))
                for k in range(0, len(sheet), 4)]
        cv2.imwrite(str(out / "contact_sheet.png"), np.vstack(rows))
    print(f"wrote {out.relative_to(REPO)}: {n} frames, labels {counts}, {empty} frames without any")
    return 0


if __name__ == "__main__":
    sys.exit(main())
