"""Run a trained model on a folder of frames.

    # classification -> predictions.csv
    python -m dino.infer --run $DINO_RUNS/<cls_run> --input <split_dir or image dir> --out <dir>

    # segmentation -> FoundationPose scene: rgb/ depth/ masks/ (binary, target object) cam_K.txt
    #                 + sem/ (full label maps) + detections.csv
    python -m dino.infer --run $DINO_RUNS/<seg_run> --input $DINO_DATA/x/val --out <scene_dir> [--target star]

Without --target, each frame's mask is the largest predicted object class in that frame.
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from .backbones import IMAGENET_MEAN, IMAGENET_STD
from .data.datasets import IMG_EXT, list_frames, read_rgb, round_to_patch
from .data.rgbd import export_frame, read_K
from .evaluate import autocast
from .model import load_run


def find_images(d):
    if os.path.isdir(os.path.join(d, "rgb")):
        return list_frames(d)
    return sorted((os.path.splitext(f)[0], os.path.join(d, f)) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", help="seg: object class to write into masks/ (default: largest per frame)")
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()

    model, cfg, classes = load_run(a.run)
    size = round_to_patch(cfg["img_size"], model.patch_size)
    tf = v2.Compose([v2.Resize(size, antialias=True), v2.ToDtype(torch.float32, scale=True),
                     v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    frames = find_images(a.input)
    os.makedirs(a.out, exist_ok=True)
    K_path = os.path.join(a.input, "cam_K.txt")
    K = read_K(K_path) if os.path.exists(K_path) else None
    target = classes.index(a.target) + 1 if a.target else None

    rows = []
    for b in range(0, len(frames), a.batch_size):
        chunk = frames[b:b + a.batch_size]
        raw = [read_rgb(p) for _, p in chunk]
        x = torch.stack([tf(r) for r in raw]).cuda()
        with autocast(cfg):
            logits = model(x).float()
        if cfg["task"] == "cls":
            prob = logits.softmax(1)
            conf, pred = prob.max(1)
            for (fid, _), c, p in zip(chunk, conf.tolist(), pred.tolist()):
                rows.append({"id": fid, "pred": classes[p], "conf": round(c, 4)})
            continue
        for (fid, path), r, lg in zip(chunk, raw, logits):
            H, W = r.shape[-2:]
            sem = F.interpolate(lg[None], size=(H, W), mode="bilinear", align_corners=False)[0].argmax(0).cpu().numpy()
            counts = np.bincount(sem.ravel(), minlength=len(classes) + 1)
            obj = target or (int(counts[1:].argmax()) + 1 if counts[1:].any() else 0)
            depth = os.path.join(a.input, "depth", fid + ".png")
            export_frame(a.out, fid, path, depth, sem == obj if obj else np.zeros_like(sem, bool), K)
            os.makedirs(os.path.join(a.out, "sem"), exist_ok=True)
            Image.fromarray(sem.astype(np.uint8)).save(os.path.join(a.out, "sem", fid + ".png"))
            rows.append({"id": fid, "object": classes[obj - 1] if obj else "none",
                         "mask_frac": round(float(counts[obj]) / sem.size, 4) if obj else 0.0})

    out_csv = os.path.join(a.out, "predictions.csv" if cfg["task"] == "cls" else "detections.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[infer] {len(rows)} frames -> {a.out}")


if __name__ == "__main__":
    main()
