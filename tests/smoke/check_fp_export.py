"""Validate a FoundationPose scene folder:  python -m tests.smoke.check_fp_export <scene_dir>"""
import os
import sys

import numpy as np
from PIL import Image


def main(d):
    K = np.loadtxt(os.path.join(d, "cam_K.txt"))
    assert K.shape == (3, 3) and K[2, 2] == 1, K
    ids = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(d, "rgb")))
    assert ids, "no frames"
    nonempty = 0
    for fid in ids:
        rgb = Image.open(os.path.join(d, "rgb", fid + ".png"))
        mask = np.asarray(Image.open(os.path.join(d, "masks", fid + ".png")))
        depth = np.asarray(Image.open(os.path.join(d, "depth", fid + ".png")))
        assert mask.shape == depth.shape == rgb.size[::-1], (fid, mask.shape, depth.shape, rgb.size)
        assert set(np.unique(mask)) <= {0, 255}, f"{fid}: mask not binary"
        assert depth.dtype == np.uint16, f"{fid}: depth dtype {depth.dtype}"
        nonempty += mask.any()
    print(f"FoundationPose export OK: {len(ids)} frames, {nonempty} non-empty masks, K=\n{K}")


if __name__ == "__main__":
    main(sys.argv[1])
