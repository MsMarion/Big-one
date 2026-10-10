"""Backbone + task head, checkpoint save/load."""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .backbones import Backbone


class ClsHead(nn.Module):
    """Linear probe on [CLS ; mean(patches)], as in the DINOv2 linear eval."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(2 * dim, affine=False)
        self.fc = nn.Linear(2 * dim, n_classes)

    def forward(self, cls, patches, out_size=None):
        return self.fc(self.bn(torch.cat([cls, patches.mean((2, 3))], 1)))


class SegHead(nn.Module):
    """Linear (BN + 1x1 conv) head on the patch grid, bilinearly upsampled to the input size."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim)
        self.conv = nn.Conv2d(dim, n_classes, 1)

    def forward(self, cls, patches, out_size):
        return F.interpolate(self.conv(self.bn(patches)), size=out_size, mode="bilinear", align_corners=False)


class DinoModel(nn.Module):
    def __init__(self, backbone, task, n_classes, unfreeze_last=0):
        super().__init__()
        self.task = task
        self.backbone = Backbone(backbone, unfreeze_last)
        head = {"cls": ClsHead, "seg": SegHead}[task]
        self.head = head(self.backbone.dim, n_classes)

    @property
    def patch_size(self):
        return self.backbone.patch_size

    def forward(self, x):
        cls, patches = self.backbone(x)
        # heads run on fp32 features (BN on bf16 is not worth the risk on ROCm)
        return self.head(cls.float(), patches.float(), x.shape[-2:])

    def trainable_state_dict(self):
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k.startswith("head.") or k in trainable}


def n_seg_classes(classes):
    return len(classes) + 1  # index 0 = background


def build_model(cfg, classes):
    n = len(classes) if cfg["task"] == "cls" else n_seg_classes(classes)
    return DinoModel(cfg["backbone"], cfg["task"], n, cfg["train"]["unfreeze_last"])


def save_checkpoint(path, model, cfg, classes, extra=None):
    torch.save({"state": model.trainable_state_dict(), "cfg": cfg, "classes": classes, **(extra or {})}, path)


def load_run(run_dir, device="cuda"):
    """Rebuild a trained model from runs/<name>/ckpt.pt. Returns (model, cfg, classes)."""
    ck = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    cfg, classes = ck["cfg"], ck["classes"]
    model = build_model(cfg, classes)
    missing, unexpected = model.load_state_dict(ck["state"], strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith("backbone.") for k in missing), [k for k in missing if not k.startswith("backbone.")]
    return model.to(device).eval(), cfg, classes


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_yaml(path, obj):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
