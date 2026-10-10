# DINO cls/seg pipeline → FoundationPose

Trains, evaluates, runs inference with and compares DINOv2 (and later DINOv3) backbones for classification and segmentation on RGB-D camera data. Everything runs on the remote AMD GPU box (`ssh dino-box`). Segmentation results are exported as FoundationPose scene folders.

## Quick start
```bash
bash scripts/sync.sh
ssh dino-box bash /workspace/dino/code/remote/bootstrap.sh     # once per container start
bash scripts/run.sh bash tests/smoke/test_pipeline.sh           # synthetic smoketest, ~4 min
bash scripts/pull.sh _compare                                   # comparison table/plot -> results/_compare
```

## Dashboard
```bash
python -m venv .venv && .venv/Scripts/pip install -r dashboard/requirements.txt
.venv/Scripts/python -m uvicorn dashboard.app:app --port 8765     # open http://127.0.0.1:8765
```
It shows live GPU/CPU/RAM/disk, processes and job logs, runs (metrics, train logs, files, mask gallery), the backbone comparison chart, and a remote FastAPI ping. Everything is pulled over SSH, because the host disables port forwarding.

## Your own data
1. Convert raw captures into the frames layout on the box:
   `python -m dino.data.rgbd --rgb_dir ... --depth_dir ... --K K.json --depth_scale 0.001 --out $DINO_DATA/mycam/train`
   (or build it locally and `bash scripts/push_data.sh ./mycam mycam`).
2. Add `masks/` (seg) and/or `labels.csv` (cls), plus `classes.txt` at the dataset root.
3. Copy `configs/finetune_seg_template.yaml`, point `data.root` at the data, then train, evaluate and infer:
   ```bash
   bash scripts/bg.sh seg1 python -m dino.train --config configs/my.yaml
   bash scripts/run.sh python -m dino.evaluate --run /workspace/dino/runs/<name>
   bash scripts/run.sh python -m dino.infer --run /workspace/dino/runs/<name> --input <frames dir> --out <scene dir> --target <class>
   ```
4. `<scene dir>` holds `rgb/ depth/ masks/ cam_K.txt`, ready for FoundationPose on an NVIDIA machine.

## Comparing backbones
Same config, different `--set backbone=dinov2_s|dinov2_b|dinov2_l|dinov3_b ...`, then
`python -m dino.compare --glob '<prefix>*'`.

## DINOv3
The weights are gated. Accept the license on HF, then either `export HF_TOKEN=...` on the box (downloads go through hf-mirror) or upload an HF-format folder to `/workspace/dino/models/dinov3_b/`. After that `--set backbone=dinov3_b` just works (patch 16; keep `img_size` a multiple of 112).

See `CLAUDE.md` for box details and conventions.
