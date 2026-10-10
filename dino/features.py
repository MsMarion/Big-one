"""Frozen-backbone feature cache.

With a frozen backbone (train.unfreeze_last == 0) the backbone output for an image never changes,
so it is computed once and heads train on the cached tensors. That removes per-epoch image decoding
(the CPU bottleneck) and the backbone forward pass.

    cls: [CLS ; mean(patches)]  -> (N, 2D) fp16
    seg: patch grid             -> (N, D, h, w) fp16, plus masks at img_size (N, H, W) uint8

Caches live under $DINO_SCRATCH/featcache (big, ephemeral disk) and are keyed by backbone, img_size,
split and a signature of the files (paths, sizes, mtimes). A container reset or any data change
just means a rebuild. Training-time augmentation is limited to horizontal flips (cached as a second view).
"""
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .backbones import resolve
from .data.datasets import build_dataset, round_to_patch

CACHE_ROOT = os.path.join(os.environ.get("DINO_SCRATCH", "/root/scratch"), "featcache")
VERSION = 1


def _signature(ds):
    h = hashlib.sha1()
    for item in ds.items:
        for p in item:
            if isinstance(p, str) and os.path.exists(p):
                st = os.stat(p)
                h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
            elif not isinstance(p, str):
                h.update(str(p).encode())  # class index
    return h.hexdigest()


def cache_key(cfg, split_key, ds, patch, flip):
    spec = {"v": VERSION, "backbone": cfg["backbone"], "src": resolve(cfg["backbone"]), "task": cfg["task"],
            "img_size": round_to_patch(cfg["img_size"], patch), "root": os.path.abspath(cfg["data"]["root"]),
            "split": cfg["data"][split_key], "flip": flip, "files": _signature(ds)}
    return hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16], spec


@torch.no_grad()
def build_or_load(cfg, split_key, model, classes, flip=False, log=print, device="cuda"):
    """Return a CachedFeatures dataset for cfg['data'][split_key], building it on a miss."""
    ds = build_dataset(cfg, split_key, False, model.patch_size, classes)  # deterministic resize, no aug
    key, spec = cache_key(cfg, split_key, ds, model.patch_size, flip)
    d = os.path.join(CACHE_ROOT, cfg["backbone"], key)
    if os.path.exists(os.path.join(d, "meta.json")):
        log(f"[cache] hit  {cfg['data'][split_key]}: {d}")
        return CachedFeatures(d), {"hit": True, "dir": d, "build_s": 0.0}

    t = time.perf_counter()
    tmp = d + f".tmp{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    views = 2 if flip else 1
    n = len(ds) * views
    bb = model.backbone
    bb.eval()
    dl = DataLoader(ds, cfg["train"]["batch_size"], num_workers=cfg["train"]["workers"], pin_memory=True)
    feats = targets = None
    i = 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["train"]["amp"] == "bf16")
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        for v in range(views):
            xv, yv = (torch.flip(x, dims=[-1]), torch.flip(y, dims=[-1]) if y.ndim == 3 else y) if v else (x, y)
            with amp:
                cls, patches = bb(xv)
            f = torch.cat([cls, patches.mean((2, 3))], 1) if cfg["task"] == "cls" else patches
            f = f.to(torch.float16).cpu().numpy()
            if feats is None:  # allocate memmaps once the shapes are known
                feats = np.lib.format.open_memmap(os.path.join(tmp, "feats.npy"), "w+", np.float16, (n,) + f.shape[1:])
                tshape = (n,) if cfg["task"] == "cls" else (n,) + tuple(yv.shape[1:])
                targets = np.lib.format.open_memmap(os.path.join(tmp, "targets.npy"), "w+",
                                                    np.int64 if cfg["task"] == "cls" else np.uint8, tshape)
            b = f.shape[0]
            feats[i:i + b] = f
            targets[i:i + b] = yv.numpy()
            i += b
    feats.flush()
    targets.flush()
    del feats, targets
    meta = {**spec, "n": n, "n_images": len(ds), "views": views, "build_s": time.perf_counter() - t,
            "out_size": round_to_patch(cfg["img_size"], model.patch_size), "created": time.time()}
    with open(os.path.join(tmp, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    if os.path.exists(d):  # another process won the race
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        os.makedirs(os.path.dirname(d), exist_ok=True)
        os.rename(tmp, d)
    size_gb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1e9
    log(f"[cache] built {cfg['data'][split_key]}: {n} samples in {meta['build_s']:.1f}s ({size_gb:.2f} GB) -> {d}")
    return CachedFeatures(d), {"hit": False, "dir": d, "build_s": meta["build_s"], "size_gb": size_gb}


class CachedFeatures(Dataset):
    """Use with num_workers=0: small caches sit in RAM, large ones are memory-mapped (never pickled to workers)."""

    RAM_LIMIT_GB = float(os.environ.get("DINO_CACHE_RAM_GB", 8))

    def __init__(self, d):
        with open(os.path.join(d, "meta.json")) as f:
            self.meta = json.load(f)
        fp = os.path.join(d, "feats.npy")
        mode = None if os.path.getsize(fp) / 1e9 < self.RAM_LIMIT_GB else "r"
        self.feats = np.load(fp, mmap_mode=mode)
        self.targets = np.load(os.path.join(d, "targets.npy"), mmap_mode=mode)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        y = self.targets[i]
        return torch.from_numpy(np.array(self.feats[i])), (int(y) if y.ndim == 0 else torch.from_numpy(y.astype(np.int64)))

    def __getitems__(self, idx):
        """Whole-batch fetch (one fancy-index per array); pair with collate_fn=batch_collate."""
        idx = np.sort(np.asarray(idx))  # sorted = sequential reads when memory-mapped; order within a batch is irrelevant
        return torch.from_numpy(np.ascontiguousarray(self.feats[idx])), torch.from_numpy(self.targets[idx].astype(np.int64))


def batch_collate(batch):
    return batch


def cache_enabled(cfg):
    c = cfg["train"].get("cache_features", "auto")
    frozen = cfg["train"]["unfreeze_last"] == 0
    if c == "auto":
        return frozen
    if c and not frozen:
        raise ValueError("train.cache_features requires a frozen backbone (train.unfreeze_last: 0)")
    return bool(c)
