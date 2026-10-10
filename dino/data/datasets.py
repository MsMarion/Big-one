"""Datasets.

Standard "frames" layout (RGB-D, FoundationPose-compatible), one folder per split or scene:

    <root>/classes.txt                 one object class per line (seg id = line index + 1, 0 = background)
    <root>/<split>/rgb/<id>.png|jpg
    <root>/<split>/depth/<id>.png      uint16, millimetres (optional)
    <root>/<split>/masks/<id>.png      uint8 semantic ids: 0 bg, k = classes[k-1], 255 ignore  (seg)
    <root>/<split>/labels.csv          id,class                                               (cls)
    <root>/<split>/cam_K.txt           3x3 intrinsics (optional)

Classification also accepts plain ImageFolder: <root>/<split>/<class>/*.jpg
"""
import csv
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

from ..backbones import IMAGENET_MEAN, IMAGENET_STD

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp")


def read_rgb(path):
    """uint8 CHW tensor. PIL, because the ROCm torchvision build here lacks libpng/libjpeg decoders."""
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)


def read_mask(path):
    return torch.from_numpy(np.asarray(Image.open(path)).copy())


def read_classes(root, split="train"):
    path = os.path.join(root, "classes.txt")
    if os.path.exists(path):
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]
    d = os.path.join(root, split)  # ImageFolder fallback
    return sorted(e for e in os.listdir(d) if os.path.isdir(os.path.join(d, e)) and e not in ("rgb", "depth", "masks"))


def list_frames(split_dir):
    rgb = os.path.join(split_dir, "rgb")
    return sorted((os.path.splitext(f)[0], os.path.join(rgb, f)) for f in os.listdir(rgb) if f.lower().endswith(IMG_EXT))


def round_to_patch(size, patch):
    return [max(patch, round(s / patch) * patch) for s in size]


def _finish():
    return [v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


class ClsDataset(Dataset):
    def __init__(self, root, split, img_size, train, classes=None):
        self.classes = classes or read_classes(root)
        idx = {c: i for i, c in enumerate(self.classes)}
        d = os.path.join(root, split)
        self.items = []
        if os.path.exists(os.path.join(d, "labels.csv")):
            frames = dict(list_frames(d))
            with open(os.path.join(d, "labels.csv")) as f:
                for row in csv.DictReader(f):
                    self.items.append((frames[row["id"]], idx[row["class"]], row["id"]))
        else:
            for c in self.classes:
                cd = os.path.join(d, c)
                for fn in sorted(os.listdir(cd)) if os.path.isdir(cd) else []:
                    if fn.lower().endswith(IMG_EXT):
                        self.items.append((os.path.join(cd, fn), idx[c], f"{c}/{os.path.splitext(fn)[0]}"))
        aug = [v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True), v2.RandomHorizontalFlip(),
               v2.ColorJitter(0.2, 0.2, 0.2)] if train else [v2.Resize(img_size, antialias=True)]
        self.tf = v2.Compose(aug + _finish())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y, _ = self.items[i]
        img = tv_tensors.Image(read_rgb(path))
        return self.tf(img), y


class SegDataset(Dataset):
    def __init__(self, root, split, img_size, train, classes=None):
        self.classes = classes or read_classes(root)
        d = os.path.join(root, split)
        self.items = [(fid, p, os.path.join(d, "masks", fid + ".png")) for fid, p in list_frames(d)]
        aug = [v2.Resize(img_size, antialias=True)]
        if train:
            aug += [v2.RandomHorizontalFlip(), v2.ColorJitter(0.2, 0.2, 0.2)]
        self.tf = v2.Compose(aug + _finish())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        _, path, mpath = self.items[i]
        img = tv_tensors.Image(read_rgb(path))
        mask = tv_tensors.Mask(read_mask(mpath))
        img, mask = self.tf(img, mask)
        return img, mask.long()


def build_dataset(cfg, split_key, train, patch_size, classes=None):
    root = cfg["data"]["root"]
    size = round_to_patch(cfg["img_size"], patch_size)
    ds = {"cls": ClsDataset, "seg": SegDataset}[cfg["task"]]
    return ds(root, cfg["data"][split_key], size, train, classes)
