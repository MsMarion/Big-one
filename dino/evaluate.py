"""Metrics + benchmarking. CLI re-evaluates a saved run (checkpoint round-trip) and adds latency:

    python -m dino.evaluate --run $DINO_RUNS/<name> [--split val]
    # benchmark a run on another dataset's test split, result to a separate file:
    python -m dino.evaluate --run $DINO_RUNS/<name> --data_root $DINO_DATA/new --split test --out eval.json
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

from .data.datasets import build_dataset, read_classes, round_to_patch
from .model import load_run, write_json

IGNORE = 255


def autocast(cfg):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"] == "bf16")


@torch.no_grad()
def confusion(model, loader, n, cfg, device="cuda", forward=None):
    conf = torch.zeros(n * n, dtype=torch.long, device=device)
    model.eval()
    forward = forward or model
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(cfg):
            pred = forward(x).argmax(1)
        keep = y != IGNORE
        conf += torch.bincount(n * y[keep] + pred[keep], minlength=n * n)
    return conf.view(n, n).cpu()


def cls_metrics(conf, classes):
    tp, support = conf.diag().float(), conf.sum(1).float()
    per = (tp / support.clamp(min=1)).tolist()
    return {"acc": (tp.sum() / support.sum()).item(), "macro_acc": sum(per) / len(per),
            "per_class_acc": dict(zip(classes, per)), "confusion": conf.tolist()}


def seg_metrics(conf, classes):
    names = ["background"] + list(classes)
    tp = conf.diag().float()
    union = conf.sum(0).float() + conf.sum(1).float() - tp
    present = union > 0
    iou = tp / union.clamp(min=1)
    fg = present.clone()
    fg[0] = False
    return {"miou": iou[present].mean().item(), "miou_fg": iou[fg].mean().item() if fg.any() else 0.0,
            "pixel_acc": (tp.sum() / conf.sum()).item(), "per_class_iou": dict(zip(names, iou.tolist()))}


def compute_metrics(model, loader, cfg, classes, device="cuda", forward=None):
    """forward: optional callable batch -> logits (e.g. the head on cached features); default model(x)."""
    if cfg["task"] == "cls":
        return cls_metrics(confusion(model, loader, len(classes), cfg, device, forward), classes)
    return seg_metrics(confusion(model, loader, len(classes) + 1, cfg, device, forward), classes)


PRIMARY = {"cls": "acc", "seg": "miou"}


@torch.no_grad()
def benchmark(model, cfg, device="cuda", batch_sizes=(1, 32), iters=20):
    size = round_to_patch(cfg["img_size"], model.patch_size)
    model.eval()
    out = {"img_size": size}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, *size, device=device)
        torch.cuda.reset_peak_memory_stats()
        with autocast(cfg):
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                model(x)
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
        out[f"bs{bs}"] = {"ms_per_batch": dt * 1e3, "imgs_per_s": bs / dt,
                          "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data_root", help="evaluate on this dataset instead of the one the run trained on")
    ap.add_argument("--out", help="write the eval JSON here instead of merging into the run's metrics.json")
    ap.add_argument("--batch_size", type=int, default=32)
    a = ap.parse_args()

    model, cfg, classes = load_run(a.run)
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    data_classes = read_classes(cfg["data"]["root"], a.split)
    if data_classes != classes:
        sys.exit(f"[eval] class mismatch: run {cfg['name']} has {classes}, dataset has {data_classes}")
    cfg["data"]["_eval"] = a.split
    ds = build_dataset(cfg, "_eval", False, model.patch_size, classes)
    dl = DataLoader(ds, a.batch_size, num_workers=cfg["train"]["workers"])
    t = time.perf_counter()
    metrics = compute_metrics(model, dl, cfg, classes)
    eval_s = time.perf_counter() - t
    bench = benchmark(model, cfg)
    key = PRIMARY[cfg["task"]]
    print(f"[eval] {cfg['name']} on {cfg['data']['root']}/{a.split} (n={len(ds)}): {key}={metrics[key]:.4f}  "
          f"bs1={bench['bs1']['ms_per_batch']:.1f}ms  bs32={bench['bs32']['imgs_per_s']:.0f} img/s")

    result = {"run": cfg["name"], "data_root": cfg["data"]["root"], "split": a.split, "n": len(ds),
              "eval_s": eval_s, "metrics": metrics, "benchmark": bench}
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        write_json(a.out, result)
        return
    mpath = os.path.join(a.run, "metrics.json")
    with open(mpath) as f:
        m = json.load(f)
    m["eval"] = result
    write_json(mpath, m)


if __name__ == "__main__":
    main()
