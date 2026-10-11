"""Convert Replicator BasicWriter output (data/raw/<name>/) into an Ultralytics YOLO-seg dataset.

    C:\\isaac-sim\\python.bat tools\\convert_to_yolo.py --raw data/raw/soup_can_v1 --out data/datasets/soup_can_v1
    (also runs in the pnp container with /repo paths)

Input per frame (formats verified on the v0 output, Replicator 1.12.27):
  rgb_NNNN.png                                   the image
  instance_segmentation_NNNN.png                 uint16 instance id per pixel
  instance_segmentation_semantics_mapping_NNNN.json   {id: {"class": label}}
  bounding_box_2d_tight_NNNN.npy (+ _prim_paths_NNNN.json)  per-instance box incl. occlusionRatio
  instance_segmentation_mapping_NNNN.json        {id: prim path}

Output (YOLO-seg): images/{train,val}/*.png, labels/{train,val}/*.txt with one line per object:
  <class_id> x1 y1 x2 y2 ...   (polygon, normalized to 0..1)
plus dataset.yaml (for `yolo train`) and manifest.json (provenance + statistics).

Why we write our own converter: Replicator's CocoWriter leaves polygons empty in this version, and keeping
the raw output means we can re-convert with different filters without re-rendering.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]


def frame_ids(raw: Path):
    return sorted(int(p.stem.split("_")[-1]) for p in raw.glob("rgb_*.png"))


def occlusion_by_path(raw: Path, i: int) -> dict:
    boxes = np.load(raw / f"bounding_box_2d_tight_{i:04d}.npy")
    paths = json.loads((raw / f"bounding_box_2d_tight_prim_paths_{i:04d}.json").read_text())
    return {p: float(b["occlusionRatio"]) for p, b in zip(paths, boxes)}


def instance_polygons(raw: Path, i: int, classes: dict, min_area: int, max_occlusion: float, eps_px: float):
    """-> list of (class_id, polygon Nx2 in pixels), plus counts of skipped instances."""
    seg = cv2.imread(str(raw / f"instance_segmentation_{i:04d}.png"), cv2.IMREAD_UNCHANGED)
    sem = json.loads((raw / f"instance_segmentation_semantics_mapping_{i:04d}.json").read_text())
    paths = json.loads((raw / f"instance_segmentation_mapping_{i:04d}.json").read_text())
    occl = occlusion_by_path(raw, i)
    out, skipped = [], {"small": 0, "occluded": 0}
    for sid, info in sem.items():
        label = info.get("class")
        if label not in classes:
            continue
        mask = (seg == int(sid)).astype(np.uint8)
        area = int(mask.sum())
        if area < min_area:
            skipped["small"] += 1
            continue
        if occl.get(paths.get(sid), 0.0) > max_occlusion:
            skipped["occluded"] += 1
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # A partly occluded object can split into pieces; YOLO-seg takes one polygon per object -> keep the
        # largest piece (the small ones are usually slivers at an occluder's edge).
        c = max(contours, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
        if len(poly) >= 3:
            out.append((classes[label], poly))
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, nargs="+", help="one or more data/raw/<name> folders")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", nargs="+", default=["soup_can"])
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=150, help="skip instances with fewer visible pixels")
    ap.add_argument("--max-occlusion", type=float, default=0.8, help="skip instances more occluded than this")
    ap.add_argument("--eps-px", type=float, default=1.0, help="polygon simplification tolerance")
    a = ap.parse_args()

    classes = {c: k for k, c in enumerate(a.classes)}
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    items = []
    for r in a.raw:
        raw = Path(r) if Path(r).is_absolute() else REPO / r
        items += [(raw, i) for i in frame_ids(raw)]
    rng = np.random.default_rng(a.seed)
    is_val = np.zeros(len(items), bool)
    is_val[rng.permutation(len(items))[: int(round(a.val_fraction * len(items)))]] = True

    stats = {"images": 0, "instances": 0, "negative_images": 0, "skipped_small": 0, "skipped_occluded": 0,
             "instance_area_px": []}
    for k, (raw, i) in enumerate(items):
        split = "val" if is_val[k] else "train"
        name = f"{raw.name}_{i:05d}"
        img = cv2.imread(str(raw / f"rgb_{i:04d}.png"))
        h, w = img.shape[:2]
        polys, skipped = instance_polygons(raw, i, classes, a.min_area, a.max_occlusion, a.eps_px)
        shutil.copyfile(raw / f"rgb_{i:04d}.png", out / "images" / split / f"{name}.png")
        lines = [f"{cid} " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in poly) for cid, poly in polys]
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats["images"] += 1
        stats["instances"] += len(polys)
        stats["negative_images"] += int(not polys)
        stats["skipped_small"] += skipped["small"]
        stats["skipped_occluded"] += skipped["occluded"]
        stats["instance_area_px"] += [int(cv2.contourArea(p.astype(np.float32))) for _, p in polys]

    areas = np.array(stats.pop("instance_area_px") or [0])
    stats["instance_area_px_p5_p50_p95"] = [int(np.percentile(areas, q)) for q in (5, 50, 95)]
    (out / "dataset.yaml").write_text(yaml.safe_dump(
        {"path": str(out), "train": "images/train", "val": "images/val", "names": {v: k for k, v in classes.items()}},
        sort_keys=False))
    sources = []
    for r in a.raw:
        raw = Path(r) if Path(r).is_absolute() else REPO / r
        cfg = yaml.safe_load((raw / "sdg_config.yaml").read_text()) if (raw / "sdg_config.yaml").exists() else {}
        sources.append({"raw": str(raw.relative_to(REPO)) if raw.is_relative_to(REPO) else str(raw),
                        "sdg_name": cfg.get("name"), "seed": cfg.get("seed")})
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sources": sources,
                "classes": a.classes, "filters": {"min_area_px": a.min_area, "max_occlusion": a.max_occlusion,
                                                  "eps_px": a.eps_px},
                "split": {"val_fraction": a.val_fraction, "seed": a.seed, "val_images": int(is_val.sum())},
                "stats": stats}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["stats"], indent=2))
    print(f"dataset written to {out}")


if __name__ == "__main__":
    main()
