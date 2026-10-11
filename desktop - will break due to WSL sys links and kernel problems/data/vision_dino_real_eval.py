"""DINOv2 identity on REAL camera frames (the wrist D405 / the D435i), training-free, no depth needed:
SAM2 automatic masks (region proposals) -> each region's gray-background crop -> DINOv2 -> the nearest mesh-render
template (models/vision/templates_mixed_v1.npz: 4 bricks + can + box). Below `--min-sim` -> "?" (unknown): the board,
the jaws, a box that isn't in the catalog should come out as "?", not as a catalog object.
    /opt/fp_venv/bin/python /repo/data/vision_dino_real_eval.py [--min-sim 0.6] -> data/vision_dino_real.png
"""
import argparse
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/repo")
from pnp_core.vision import crop_for_embedding  # noqa: E402

FRAMES = [  # (file, crop of the composite: x0, x1) -- the saved composites have the wrist image on the left
    ("data/real_frames/wrist_color.png", None), ("data/real_frames/fixed_color.png", None),
    ("data/jetson_brick_calibrated.png", (0, 640)), ("data/jetson_hover.png", (0, 640)),
    ("data/jetson_first_brick.png", (0, 640)), ("data/jetson_first_grab.png", (0, 640)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sim", type=float, default=0.6)
    ap.add_argument("--max-area", type=float, default=0.15)
    a = ap.parse_args()
    import torch
    from transformers import AutoImageProcessor, AutoModel, pipeline
    gen = pipeline("mask-generation", model="/repo/models/vision/sam2.1-hiera-tiny", device=0)
    proc = AutoImageProcessor.from_pretrained("/repo/models/vision/dinov2-small")
    dino = AutoModel.from_pretrained("/repo/models/vision/dinov2-small").cuda().eval()
    lib = np.load("/repo/models/vision/templates_mixed_v1.npz", allow_pickle=True)
    names, E = np.array(lib["names"]), lib["embeddings"].astype(np.float32)
    tiles = []
    for path, cut in FRAMES:
        bgr = cv2.imread("/repo/" + path)
        if cut:
            bgr = bgr[:, cut[0]:cut[1]]
        rgb = bgr[..., ::-1].copy()
        out = gen(Image.fromarray(rgb), points_per_batch=64)
        masks = [np.asarray(m, bool) for m in out["masks"]]
        h, w = rgb.shape[:2]
        masks = [m for m in masks if 150 <= m.sum() <= a.max_area * h * w]
        crops = [crop_for_embedding(rgb, m) for m in masks]
        keep = [i for i, c in enumerate(crops) if c is not None]
        vis = bgr.copy()
        lines = []
        if keep:
            with torch.no_grad():
                x = proc(images=[crops[i] for i in keep], return_tensors="pt").to("cuda")
                e = dino(**x).last_hidden_state[:, 0].float().cpu().numpy()
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            S = e @ E.T
            for j, i in enumerate(keep):
                k = int(S[j].argmax())
                sim, name = float(S[j, k]), str(names[k])
                known = sim >= a.min_sim
                col = (0, 220, 0) if known and name.startswith("lego_") else (0, 200, 255) if known else (90, 90, 90)
                cs, _ = cv2.findContours(masks[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cs, -1, col, 2 if known else 1)
                ys, xs = np.nonzero(masks[i])
                label = f"{name.replace('lego_', '')} {sim:.2f}" if known else f"? {sim:.2f}"
                cv2.putText(vis, label, (int(xs.mean()) - 20, int(ys.mean())), 0, 0.42, col, 1, cv2.LINE_AA)
                lines.append((sim, name, int(masks[i].sum()), known))
        lines.sort(reverse=True)
        print(f"== {path}: {len(masks)} regions; top: " +
              "; ".join(f"{n}{'' if k else '(?)'} {s:.2f} [{ar} px]" for s, n, ar, k in lines[:6]))
        vis = cv2.resize(vis, (640, int(vis.shape[0] * 640 / vis.shape[1])))
        cv2.putText(vis, path.split("/")[-1], (6, 16), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(vis)
    hmax = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((hmax - t.shape[0], 640, 3), np.uint8)]) for t in tiles]
    sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
    cv2.imwrite("/repo/data/vision_dino_real.png", sheet)
    print("wrote data/vision_dino_real.png (green: a brick; orange: another catalog object; gray: '?')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
