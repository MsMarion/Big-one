"""Build a DINO reference library (the `dino_templates` identity backend, pnp_core/vision.py) from images per object.

    python3 tools/build_templates.py data/templates/ --out models/vision/templates_v1.npz \\
        [--model models/vision/dinov2-small] [--device cuda]

data/templates/<catalog name>/*.png|jpg   one folder per object: mesh renders, photos (web or the robot's), or the
                                          crops the robot saved of objects it identified (background gray or plain)
-> .npz with `names` (one per image) and `embeddings` (L2-normalised DINO CLS tokens). Adding an object = adding a
folder and re-running: seconds, no training. The same model directory must be used at run time.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="models/vision/dinov2-small")
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    import torch
    from transformers import AutoImageProcessor, AutoModel
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc, model = AutoImageProcessor.from_pretrained(a.model), AutoModel.from_pretrained(a.model).to(dev).eval()
    names, paths = [], []
    for d in sorted(p for p in Path(a.folder).iterdir() if p.is_dir()):
        imgs = sorted(q for q in d.iterdir() if q.suffix.lower() in (".png", ".jpg", ".jpeg"))
        names += [d.name] * len(imgs)
        paths += imgs
        print(f"  {d.name}: {len(imgs)} images")
    if not paths:
        sys.exit(f"no images under {a.folder}/<name>/")
    E = []
    with torch.no_grad():
        for i in range(0, len(paths), a.batch):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + a.batch]]
            x = proc(images=ims, return_tensors="pt").to(dev)
            e = model(**x).last_hidden_state[:, 0].float().cpu().numpy()
            E.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    E = np.concatenate(E)
    np.savez(a.out, names=np.array(names), embeddings=E.astype(np.float16), model=a.model)
    # how separable are they? the nearest OTHER object's similarity, per object (lower = easier)
    for n in sorted(set(names)):
        own = E[np.array(names) == n]
        other = E[np.array(names) != n]
        if len(other):
            print(f"  {n}: closest other object at cosine {float((own @ other.T).max()):.2f}")
    print(f"wrote {a.out}: {len(names)} templates, {len(set(names))} objects, dim {E.shape[1]} ({dev})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
