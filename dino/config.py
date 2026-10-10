"""YAML config loading with `--set a.b=value` overrides and $ENV expansion."""
import copy
import os

import yaml

DEFAULTS = {
    "name": None,
    "task": "cls",  # cls | seg
    "backbone": "dinov2_s",
    "seed": 0,
    "img_size": [224, 224],  # (H, W); multiples of 112 work for both patch 14 (v2) and 16 (v3)
    "data": {"root": "$DINO_DATA/synthetic", "train": "train", "val": "val"},
    "train": {
        "epochs": 5,
        "batch_size": 32,
        "lr": 1e-3,
        "backbone_lr_mult": 0.1,  # only used when unfreeze_last > 0
        "weight_decay": 1e-4,
        "warmup_frac": 0.05,
        "unfreeze_last": 0,  # 0 = frozen backbone (linear probe); N = finetune last N blocks
        "cache_features": "auto",  # auto = on when frozen: backbone runs once, heads train on cached features
        "cache_flip": True,  # with the cache, augmentation = cached horizontal flip only
        "workers": 4,
        "amp": "bf16",  # bf16 | fp32
        "log_every": 20,
    },
}


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _set(d, dotted, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _expand(x):
    if isinstance(x, str):
        return os.path.expandvars(x)
    if isinstance(x, dict):
        return {k: _expand(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_expand(v) for v in x]
    return x


def load_config(path=None, overrides=()):
    cfg = {}
    if path:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, cfg)
    for o in overrides or ():
        k, v = o.split("=", 1)
        _set(cfg, k, yaml.safe_load(v))
    cfg = _expand(cfg)
    if not cfg["name"]:
        cfg["name"] = f"{cfg['task']}_{cfg['backbone']}"
    return cfg


def runs_root():
    return os.environ.get("DINO_RUNS", os.path.abspath("runs"))
