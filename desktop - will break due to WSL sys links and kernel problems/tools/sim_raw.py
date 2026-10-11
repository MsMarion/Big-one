"""Read Isaac Replicator raw output (sim/sdg/generate_dataset.py: BasicWriter files + frames.jsonl) as RGB-D Frames
(pnp_core/vision.py) with the ground truth: for the vision evaluations and the auto-labeller.

    src = SimRaw("data/raw/lego_test");  f = src.frame(0);  gt = src.truth(0)   # [(class, mask, xy, yaw)]
"""
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from pnp_core.vision import Frame

USD_TO_OPTICAL = np.diag([1.0, -1.0, -1.0, 1.0])    # USD camera (y up, -z forward) -> OpenCV optical


class SimRaw:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.frames = [json.loads(line) for line in (self.dir / "frames.jsonl").read_text().split("\n") if line.strip()]
        cfg = yaml.safe_load((self.dir / "sdg_config.yaml").read_text())
        repo = Path(__file__).resolve().parents[1]
        scene = yaml.safe_load((repo / cfg["scene"]).read_text())
        t = scene["table"]
        self.table_z = t["center"][2] + t["size"][2] / 2
        r = cfg["targets"]["region"]
        self.region = (r["x"][0], r["x"][1], r["y"][0], r["y"][1])

    def __len__(self):
        return len(self.frames)

    def frame(self, i: int) -> Frame:
        c = json.loads((self.dir / f"camera_params_{i:04d}.json").read_text())
        T = np.linalg.inv(np.array(c["cameraViewTransform"]).reshape(4, 4).T) @ USD_TO_OPTICAL
        w, h = c["renderProductResolution"]
        P = np.array(c["cameraProjection"]).reshape(4, 4)
        K = np.array([[P[0, 0] * w / 2, 0, w / 2], [0, P[1, 1] * h / 2, h / 2], [0, 0, 1.0]])
        d = np.load(self.dir / f"distance_to_image_plane_{i:04d}.npy").astype(np.float64)
        d[~np.isfinite(d)] = 0.0
        rgb = np.array(Image.open(self.dir / f"rgb_{i:04d}.png").convert("RGB"))
        return Frame(rgb, d, K, T, camera="sim")

    def truth(self, i: int):
        """-> [(class, instance mask)] of the labelled objects in the image (exact, from the instance segmentation)."""
        seg = np.array(Image.open(self.dir / f"instance_segmentation_{i:04d}.png"))
        sem = json.loads((self.dir / f"instance_segmentation_semantics_mapping_{i:04d}.json").read_text())
        out = []
        for k, v in sem.items():
            cls = v.get("class", "")
            if cls in ("BACKGROUND", "UNLABELLED"):
                continue
            m = seg == int(k)
            if m.any():
                out.append((cls, m))
        return out
