"""Draw the YOLO-seg labels onto images and save a contact sheet: ALWAYS look at your training data.

    C:\\isaac-sim\\python.bat tools\\inspect_dataset.py data/datasets/soup_can_v0 [--n 16]
Writes <dataset>/inspect.png. Green outline = labeled soup can. Images without labels get a red border
(negatives: check there really is no fully visible can in them).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=4)
    a = ap.parse_args()
    ds = Path(a.dataset) if Path(a.dataset).is_absolute() else REPO / a.dataset
    imgs = sorted((ds / "images" / a.split).glob("*.png"))
    pick = np.random.default_rng(a.seed).permutation(len(imgs))[: a.n]
    tiles = []
    for k in pick:
        img = cv2.imread(str(imgs[k]))
        h, w = img.shape[:2]
        lines = (ds / "labels" / a.split / (imgs[k].stem + ".txt")).read_text().split("\n")
        n = 0
        for line in filter(None, lines):
            v = np.array(line.split()[1:], float).reshape(-1, 2) * [w, h]
            cv2.polylines(img, [v.astype(np.int32)], True, (0, 255, 0), 2)
            n += 1
        if n == 0:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.putText(img, f"{imgs[k].stem}  cans={n}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(img, f"{imgs[k].stem}  cans={n}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(cv2.resize(img, (w // 2, h // 2)))
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r:r + a.cols]) for r in range(0, len(tiles), a.cols)]
    out = ds / "inspect.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"contact sheet: {out}")


if __name__ == "__main__":
    main()
