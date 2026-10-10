"""DINO backbones behind one interface: forward(x) -> (cls [B,D], patches [B,D,h,w]).

Weights load through HF transformers. On the box HF_ENDPOINT points at hf-mirror.com because
huggingface.co is blocked. A folder at $DINO_MODELS/<name>/ (HF format) takes precedence over
the hub id, which is how gated DINOv3 weights can be supplied by manual upload.
"""
import os

import torch
import torch.nn as nn
from transformers import AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

REGISTRY = {
    "dinov2_s": "facebook/dinov2-small",
    "dinov2_b": "facebook/dinov2-base",
    "dinov2_l": "facebook/dinov2-large",
    "dinov2_g": "facebook/dinov2-giant",
    "dinov2_s_reg": "facebook/dinov2-with-registers-small",
    "dinov2_b_reg": "facebook/dinov2-with-registers-base",
    # DINOv3 is gated: accept the license on HF and export HF_TOKEN, or upload to $DINO_MODELS/<name>/
    "dinov3_s": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_b": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_l": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def resolve(name):
    local = os.path.join(os.environ.get("DINO_MODELS", ""), name)
    if os.environ.get("DINO_MODELS") and os.path.isdir(local):
        return local
    if name in REGISTRY:
        return REGISTRY[name]
    if os.path.isdir(name) or "/" in name:  # raw path or HF repo id
        return name
    raise KeyError(f"unknown backbone {name!r}; known: {sorted(REGISTRY)}")


def _find_blocks(model, n_layers):
    for m in model.modules():
        if isinstance(m, nn.ModuleList) and len(m) == n_layers:
            return m
    raise RuntimeError("could not locate transformer blocks")


class Backbone(nn.Module):
    def __init__(self, name, unfreeze_last=0):
        super().__init__()
        self.name = name
        self.model = AutoModel.from_pretrained(resolve(name))
        cfg = self.model.config
        self.patch_size = cfg.patch_size
        self.dim = cfg.hidden_size
        self.n_prefix = 1 + getattr(cfg, "num_register_tokens", 0)  # CLS + registers
        self.frozen = unfreeze_last == 0

        self.requires_grad_(False)
        if unfreeze_last > 0:
            for blk in _find_blocks(self.model, cfg.num_hidden_layers)[-unfreeze_last:]:
                blk.requires_grad_(True)
            for attr in ("layernorm", "norm"):  # final norm: Dinov2Model / DINOv3ViTModel
                if isinstance(getattr(self.model, attr, None), nn.Module):
                    getattr(self.model, attr).requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, x):
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            tok = self.model(pixel_values=x).last_hidden_state
        cls = tok[:, 0]
        patches = tok[:, self.n_prefix:]
        assert patches.shape[1] == h * w, f"got {patches.shape[1]} patch tokens, expected {h}x{w}"
        return cls, patches.transpose(1, 2).reshape(B, self.dim, h, w)
