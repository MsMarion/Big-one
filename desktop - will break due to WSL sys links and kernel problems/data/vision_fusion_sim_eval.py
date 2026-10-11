"""Geometry vs DINO vs their fusion (pnp_core/vision.py) on the SIM brick test set (data/raw/lego_test: Isaac RGB-D,
unseen seed, ground truth in frames.jsonl, distractor obstacles in the scene). Every pipeline sees the same frames.
    /opt/fp_venv/bin/python /repo/data/vision_fusion_sim_eval.py [--frames 300]   (cwd /repo)
Scored per ground-truth brick (matched within 15 mm): right type / wrong type / missed; plus false positives (an
estimate with no brick within 15 mm: a distractor, the arm, a ghost).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/repo")
from pnp_core.catalog import load_yaml  # noqa: E402
from pnp_core.vision import Frame, build_pipeline  # noqa: E402

RAW = Path("/repo/data/raw/lego_test")
TABLE_Z = -0.0024                                   # config/scenes/so101_lego.yaml
REGION = (0.06, 0.34, -0.20, 0.20)                  # the bricks' spawn region (sdg_config.yaml)
USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])    # USD camera (y up, -z forward) -> OpenCV optical


def frame(i: int) -> Frame:
    c = json.loads((RAW / f"camera_params_{i:04d}.json").read_text())
    T = np.linalg.inv(np.array(c["cameraViewTransform"]).reshape(4, 4).T) @ USD_TO_OPTICAL
    w, h = c["renderProductResolution"]
    P = np.array(c["cameraProjection"]).reshape(4, 4)
    K = np.array([[P[0, 0] * w / 2, 0, w / 2], [0, P[1, 1] * h / 2, h / 2], [0, 0, 1.0]])
    d = np.load(RAW / f"distance_to_image_plane_{i:04d}.npy").astype(np.float64)
    d[~np.isfinite(d)] = 0.0
    rgb = np.array(Image.open(RAW / f"rgb_{i:04d}.png").convert("RGB"))
    return Frame(rgb, d, K, T, camera="sim_d435i")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    a = ap.parse_args()
    cat = load_yaml("config/objects.yaml")
    prof = load_yaml("config/vision.yaml")["profiles"]
    lib = {"library": "models/vision/templates_mixed_v1.npz", "model_dir": "models/vision/dinov2-small"}
    pipes = {
        "geometry (depth + catalog size)": build_pipeline(prof["geometry"], cat, TABLE_Z, REGION),
        "DINO only (depth blobs, DINO names them)": build_pipeline(
            {"proposals": [{"name": "depth_blobs"}],
             "identity": [{"name": "dino_templates", **lib, "min_score": 0.5, "margin": 0.0}],
             "pose": [{"name": "box_geometry"}]}, cat, TABLE_Z, REGION),
        "fusion (size names it, DINO checks the kind)": build_pipeline(prof["fusion"], cat, TABLE_Z, REGION),
    }
    gt = [json.loads(line) for line in (RAW / "frames.jsonl").read_text().split("\n") if line.strip()]
    stats = {k: dict(right=0, wrong=0, missed=0, fp=0, fp_distractor=0, t=0.0, why={}) for k in pipes}

    def why(name, xy, bricks, f, e=None):
        """a reason for a miss / wrong type: a neighbour within 2.5 cm (touching -> merged blob), at the edge of the
        region, or the predicted type"""
        close = [n for n, q in bricks if 0 < np.linalg.norm(q - xy) < 0.025]
        edge = not (REGION[0] + 0.012 < xy[0] < REGION[1] - 0.012 and REGION[2] + 0.012 < xy[1] < REGION[3] - 0.012)
        tag = "touching another brick" if close else "region edge" if edge else "isolated"
        return tag + (f", seen as {e.identity.name.replace('lego_', '')} "
                      f"{e.proposal.footprint[0] * 1000:.0f}x{e.proposal.footprint[1] * 1000:.0f}x"
                      f"{e.proposal.height * 1000:.0f}" if e is not None and tag == "isolated" else "")
    n_bricks = 0
    for i in range(min(a.frames, len(gt))):
        f = frame(i)
        bricks = [(o["label"], np.array(o["xy"])) for o in gt[i]["objects"] if (o.get("label") or "").startswith("lego_")]
        others = [np.array(o["xy"]) for o in gt[i]["objects"] if not (o.get("label") or "").startswith("lego_")]
        n_bricks += len(bricks)
        for k, pipe in pipes.items():
            t0 = time.time()
            est = pipe.run(f)
            stats[k]["t"] += time.time() - t0
            used = set()
            for name, xy in bricks:
                cand = [(np.linalg.norm(np.array(e.proposal.center_xy) - xy), j) for j, e in enumerate(est)
                        if j not in used and e.proposal.center_xy is not None]
                cand = [c for c in cand if c[0] < 0.015]
                if not cand:
                    stats[k]["missed"] += 1
                    r = "missed: " + why(name, xy, bricks, f)
                    stats[k]["why"][r] = stats[k]["why"].get(r, 0) + 1
                    continue
                _, j = min(cand)
                used.add(j)
                ok = est[j].identity.name == name
                stats[k]["right" if ok else "wrong"] += 1
                if not ok:
                    r = f"wrong ({name.replace('lego_', '')}): " + why(name, xy, bricks, f, est[j])
                    r = r.split(", seen as")[0] if "isolated" not in r else r[:60]
                    stats[k]["why"][r] = stats[k]["why"].get(r, 0) + 1
            for j, e in enumerate(est):
                if j in used:
                    continue
                stats[k]["fp"] += 1
                if any(np.linalg.norm(np.array(e.proposal.center_xy) - xy) < 0.04 for xy in others):
                    stats[k]["fp_distractor"] += 1
    print(f"{min(a.frames, len(gt))} frames, {n_bricks} ground-truth bricks (some hidden by the arm or out of view)")
    for k, s in stats.items():
        print(f"\n== {k}")
        print(f"  right type {s['right']} ({s['right'] / n_bricks * 100:.1f}%), wrong type {s['wrong']}, "
              f"missed {s['missed']}")
        print(f"  false positives {s['fp']} (on a distractor obstacle: {s['fp_distractor']}); "
              f"{s['t'] / min(a.frames, len(gt)) * 1000:.0f} ms per frame")
        for r, c in sorted(s["why"].items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {c:4d}  {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
