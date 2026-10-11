"""DINOv2 identity on the SIM test sets (lego_test, ycb2_test: Isaac Replicator renders, unseen seed, exact masks).
Every labelled instance -> its mask -> the gray-background crop (pnp_core.vision.crop_for_embedding) -> DINOv2 CLS
-> nearest reference. Two reference libraries:
  A. mesh renders (tools/render_templates.py, models/vision/templates_mixed_v1.npz): NO sim image used -- the
     "add an object from its mesh" path; the 2x3 (never in the sim data) is a decoy
  B. k crops per class from the TRAINING sim sets (lego_v1, ycb2_v1): the "a few example images" path
    /opt/fp_venv/bin/python /repo/data/vision_dino_sim_eval.py [--k 20] [--max-images 300]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/repo")
from pnp_core.vision import crop_for_embedding  # noqa: E402

DS = Path("/repo/data/datasets")
CLASS_TO_CAT = {"lego_2x4": "lego_2x4", "lego_2x2": "lego_2x2", "lego_1x2": "lego_1x2",
                "soup_can": "tomato_soup_can", "cracker_box": "cracker_box"}


def kind(n):
    return "brick" if n.startswith("lego_") else n


def instances(name, split, max_images, rng=None, per_class=None):
    """-> list of (catalog name, crop, mask area px)."""
    import yaml
    names = yaml.safe_load((DS / name / "dataset.yaml").read_text())["names"]
    imgs = sorted((DS / name / "images" / split).glob("*.png")) + sorted((DS / name / "images" / split).glob("*.jpg"))
    if rng is not None:
        rng.shuffle(imgs)
    out, count = [], {}
    for p in imgs[:max_images]:
        lab = DS / name / "labels" / split / (p.stem + ".txt")
        if not lab.exists():
            continue
        im = np.array(Image.open(p).convert("RGB"))
        h, w = im.shape[:2]
        for line in lab.read_text().split("\n"):
            v = line.split()
            if len(v) < 7:
                continue
            cat = CLASS_TO_CAT[names[int(v[0])]]
            if per_class and count.get(cat, 0) >= per_class:
                continue
            xy = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).ravel().tolist()
            m = Image.new("L", (w, h), 0)
            ImageDraw.Draw(m).polygon(xy, fill=1)
            mask = np.array(m, bool)
            c = crop_for_embedding(im, mask)
            if c is not None:
                out.append((cat, c, int(mask.sum())))
                count[cat] = count.get(cat, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=300)
    ap.add_argument("--model", default="/repo/models/vision/dinov2-small")
    a = ap.parse_args()
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc, model = AutoImageProcessor.from_pretrained(a.model), AutoModel.from_pretrained(a.model).cuda().eval()

    def embed(crops):
        E = []
        with torch.no_grad():
            for i in range(0, len(crops), 64):
                x = proc(images=crops[i:i + 64], return_tensors="pt").to("cuda")
                e = model(**x).last_hidden_state[:, 0].float().cpu().numpy()
                E.append(e / np.linalg.norm(e, axis=1, keepdims=True))
        return np.concatenate(E)

    test = instances("lego_test", "val", a.max_images) + instances("ycb2_test", "val", a.max_images)
    truth = np.array([t[0] for t in test])
    area = np.array([t[2] for t in test])
    Et = embed([t[1] for t in test])
    print(f"test: {len(test)} instances  " + ", ".join(f"{c} {np.sum(truth == c)}" for c in sorted(set(truth))))

    lib = np.load("/repo/models/vision/templates_mixed_v1.npz", allow_pickle=True)
    rng = np.random.default_rng(0)
    few = instances("lego_v1", "train", 400, rng, a.k) + instances("ycb2_v1", "train", 400, rng, a.k)
    libs = {"A mesh renders (no sim images)": (np.array(lib["names"]), lib["embeddings"].astype(np.float32)),
            f"B {a.k} sim crops per class": (np.array([f[0] for f in few]), embed([f[1] for f in few]))}
    bins = [(0, 400), (400, 1500), (1500, 10 ** 9)]
    for title, (ln, LE) in libs.items():
        pred = ln[(Et @ LE.T).argmax(axis=1)]
        kt, kp = np.array([kind(n) for n in truth]), np.array([kind(n) for n in pred])
        print(f"\n== {title}: {len(ln)} references")
        print(f"  KIND (brick / can / box): {np.mean(kp == kt) * 100:.1f}%    EXACT object: {np.mean(pred == truth) * 100:.1f}%")
        for c in sorted(set(truth)):
            s = truth == c
            wrong = {}
            for p in pred[s & (pred != truth)]:
                wrong[p] = wrong.get(p, 0) + 1
            print(f"    {c:16s} exact {np.mean(pred[s] == c) * 100:5.1f}%  kind {np.mean(kp[s] == kind(c)) * 100:5.1f}%"
                  f"   mistaken for: {dict(sorted(wrong.items(), key=lambda kv: -kv[1])) or '-'}")
        for lo, hi in bins:
            s = (area >= lo) & (area < hi)
            if s.any():
                print(f"    mask {lo:>5d}-{hi if hi < 10 ** 9 else 'inf':>5} px: kind {np.mean(kp[s] == kt[s]) * 100:5.1f}%"
                      f", exact {np.mean(pred[s] == truth[s]) * 100:5.1f}%  ({s.sum()} instances)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
