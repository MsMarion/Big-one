"""Train a YOLO11-seg detector on a converted synthetic dataset, evaluate it on the held-out test set,
and write the model + a model card (provenance + metrics) into models/.

Run in the pnp container (GPU):
    python3 /repo/tools/train_yolo.py --dataset soup_can_v1 --test soup_can_test --name soup_can_seg_v1

What this teaches:
  - Transfer learning: start from COCO-pretrained yolo11n-seg weights (downloaded once by Ultralytics),
    fine-tune on our synthetic images. Small model ("n") = fast enough for 5+ Hz on any GPU.
  - Three numbers, three questions:  val mAP = "did training converge?" (images it was tuned on, same seed
    family), TEST mAP = "does it generalize?" (different seed, never seen), and later the 3D error in the
    closed-loop sim (tools/eval_perception.py) = "is it good enough to GRASP?".
  - Ultralytics' own augmentation (HSV, blur, mosaic, flips, scale) adds image-level variety on top of
    the scene-level randomization done in Replicator.
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("PNP_DATA", "/data")) / "yolo_runs"


def local_dataset_yaml(name: str, all_as_val: bool = False) -> Path:
    """dataset.yaml was written on Windows (absolute Windows path) -> rewrite `path` for this machine."""
    ds = REPO / "data" / "datasets" / name
    cfg = yaml.safe_load((ds / "dataset.yaml").read_text())
    cfg["path"] = str(ds)
    if all_as_val:
        cfg["train"] = cfg["val"] = "images/val"
    out = ds / "dataset_local.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--test", default=None, help="held-out dataset (converted with --val-fraction 1.0)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=20)
    a = ap.parse_args()

    os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNS / "ultralytics_cfg"))
    import ultralytics
    from ultralytics import YOLO

    RUNS.mkdir(parents=True, exist_ok=True)
    data = local_dataset_yaml(a.dataset)
    t0 = time.time()
    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=a.patience,
                device=0, workers=8, cache="ram", seed=0, deterministic=False,
                project=str(RUNS), name=a.name, exist_ok=True, plots=True, verbose=False)
    train_min = (time.time() - t0) / 60
    run_dir = RUNS / a.name
    best = run_dir / "weights" / "best.pt"

    def metrics(res):
        return {"mask_mAP50": round(float(res.seg.map50), 4), "mask_mAP50_95": round(float(res.seg.map), 4),
                "box_mAP50_95": round(float(res.box.map), 4), "precision": round(float(res.seg.mp), 4),
                "recall": round(float(res.seg.mr), 4)}

    trained = YOLO(str(best))
    val = metrics(trained.val(data=str(data), imgsz=a.imgsz, device=0, verbose=False, plots=False))
    test = None
    if a.test:
        test = metrics(trained.val(data=str(local_dataset_yaml(a.test, all_as_val=True)), imgsz=a.imgsz,
                                   device=0, verbose=False, plots=False, split="val"))

    models = REPO / "models"
    models.mkdir(exist_ok=True)
    shutil.copyfile(best, models / f"{a.name}.pt")
    manifest = json.loads((REPO / "data" / "datasets" / a.dataset / "manifest.json").read_text())
    card = {
        "name": a.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "weights": f"models/{a.name}.pt", "base_model": a.model, "ultralytics": ultralytics.__version__,
        "classes": manifest["classes"],
        "train": {"dataset": a.dataset, "dataset_sources": manifest["sources"], "dataset_stats": manifest["stats"],
                  "epochs_requested": a.epochs, "imgsz": a.imgsz, "batch": a.batch, "minutes": round(train_min, 1),
                  "run_dir": str(run_dir)},
        "metrics": {"val": val, "test": test, "test_dataset": a.test},
        "gate": {"test_mask_mAP50_95_min": 0.8,
                 "passed": bool(test and test["mask_mAP50_95"] >= 0.8)},
    }
    (models / f"{a.name}.model_card.yaml").write_text(yaml.safe_dump(card, sort_keys=False))
    print(yaml.safe_dump(card["metrics"], sort_keys=False))
    print(f"model: models/{a.name}.pt  card: models/{a.name}.model_card.yaml  gate passed: {card['gate']['passed']}")


if __name__ == "__main__":
    main()
