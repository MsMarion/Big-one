"""Train a head (linear probe) or partially finetune a DINO backbone.

    python -m dino.train --config configs/smoke_cls.yaml [--set backbone=dinov2_b train.epochs=3 name=foo]

Writes $DINO_RUNS/<name>/{config.yaml, log.txt, ckpt.pt (best), metrics.json}.
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import load_config, runs_root
from .data.datasets import build_dataset, read_classes, round_to_patch
from .features import batch_collate, build_or_load, cache_enabled
from .evaluate import IGNORE, PRIMARY, autocast, compute_metrics
from .model import build_model, save_checkpoint, write_json, write_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=3")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    tc = cfg["train"]
    torch.manual_seed(cfg["seed"])
    device = "cuda"

    run_dir = os.path.join(runs_root(), cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    write_yaml(os.path.join(run_dir, "config.yaml"), cfg)
    logf = open(os.path.join(run_dir, "log.txt"), "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    classes = read_classes(cfg["data"]["root"], cfg["data"]["train"])
    model = build_model(cfg, classes).to(device)
    ps = model.patch_size
    use_cache = cache_enabled(cfg)
    cache_info = None
    if use_cache:
        # backbone runs once per (backbone, split, img_size, files); heads then train on cached features
        train_ds, ci_tr = build_or_load(cfg, "train", model, classes, flip=tc.get("cache_flip", True), log=log)
        val_ds, ci_va = build_or_load(cfg, "val", model, classes, flip=False, log=log)
        cache_info = {"train": ci_tr, "val": ci_va}
        out_size = round_to_patch(cfg["img_size"], ps)
        forward = lambda f: model.forward_features(f, out_size)
        workers = 0  # cached tensors are in RAM / memory-mapped; no decoding left to parallelize
        collate = batch_collate  # CachedFeatures.__getitems__ already returns stacked batches
    else:
        train_ds = build_dataset(cfg, "train", True, ps, classes)
        val_ds = build_dataset(cfg, "val", False, ps, classes)
        forward = model
        workers = tc["workers"]
        collate = None
    bs = min(tc["batch_size"], len(train_ds))
    train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True, num_workers=workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=workers > 0)
    val_dl = DataLoader(val_ds, tc["batch_size"], num_workers=workers, collate_fn=collate, pin_memory=True)

    head_p = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("head.")]
    bb_p = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [{"params": head_p, "lr": tc["lr"]}]
    if bb_p:
        groups.append({"params": bb_p, "lr": tc["lr"] * tc["backbone_lr_mult"]})
    opt = torch.optim.AdamW(groups, weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(train_dl)
    warm = max(1, int(total * tc["warmup_frac"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)

    n_train = sum(p.numel() for p in head_p + bb_p)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"[train] {cfg['name']}: task={cfg['task']} backbone={cfg['backbone']} patch={ps} "
        f"classes={len(classes)} train={len(train_ds)} val={len(val_ds)} "
        f"trainable={n_train/1e6:.3f}M / {n_total/1e6:.1f}M  cache={'on' if use_cache else 'off'}  "
        f"device={torch.cuda.get_device_name()}")

    key = PRIMARY[cfg["task"]]
    best, history, step = -1.0, [], 0
    torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter() - sum(c["build_s"] for c in (cache_info or {}).values())  # count cache build
    imgs_seen, t_train = 0, 0.0
    for epoch in range(tc["epochs"]):
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_loss = 0.0
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(cfg):
                loss = loss_fn(forward(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item()
            imgs_seen += x.shape[0]
            if step % tc["log_every"] == 0:
                log(f"  ep {epoch} it {i+1}/{len(train_dl)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}")
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0

        m = compute_metrics(model, val_dl, cfg, classes, device, forward)
        ep = {"epoch": epoch, "train_loss": run_loss / len(train_dl), **{k: v for k, v in m.items() if isinstance(v, float)}}
        history.append(ep)
        tag = ""
        if m[key] > best:
            best, best_m = m[key], m
            save_checkpoint(os.path.join(run_dir, "ckpt.pt"), model, cfg, classes, {"epoch": epoch, key: best})
            tag = "  *best*"
        log(f"[epoch {epoch}] loss {ep['train_loss']:.4f}  val {key} {m[key]:.4f}{tag}")

    out = {
        "name": cfg["name"], "task": cfg["task"], "backbone": cfg["backbone"], "unfreeze_last": tc["unfreeze_last"],
        "img_size": cfg["img_size"], "epochs": tc["epochs"], "primary_metric": key, "best": best, "best_metrics": best_m,
        "history": history, "train_imgs_per_s": imgs_seen / max(t_train, 1e-9),
        "train_time_s": time.perf_counter() - t_start, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "trainable_params": n_train, "total_params": n_total, "n_train": len(train_ds), "n_val": len(val_ds),
        "cache": cache_info,
    }
    write_json(os.path.join(run_dir, "metrics.json"), out)
    log(f"[done] best val {key} {best:.4f}  ({out['train_imgs_per_s']:.0f} img/s train, "
        f"peak {out['peak_vram_gb']:.1f} GB) -> {run_dir}")


if __name__ == "__main__":
    main()
