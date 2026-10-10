# DINO pipeline (remote AMD GPU box)

Classification + segmentation on DINOv2 (DINOv3 ready, pending gated access) for RGB-D camera data.
Segmentation output is exported as NVIDIA FoundationPose scene folders. FoundationPose itself needs
CUDA, so it runs on a separate NVIDIA machine and not on this box.

## Rules
- **Never run commands starting with `sudo`**, and don't apt-install anything. Python packages go into `/workspace/dino/venv` via `remote/requirements.txt` only.
- Don't use the ssh-mcp `privileged-command` tool.

## Reaching the box
- SSH alias: `ssh dino-box` (root@36.150.116.206:36168, key `~/.ssh/id_ed25519`).
- MCP server: `dino-box` in `.mcp.json` (npm `ssh-mcp`, workdir `/workspace/dino/code`, policy group `dev`).
- **Every remote command must start with `source /workspace/dino/code/remote/env.sh`**. It sets the egress proxy (non-interactive SSH skips `~/.bashrc`), `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`, the venv and the `DINO_*` paths. `scripts/run.sh` does this for you.

## Workflow (local Git Bash, from repo root)
```
bash scripts/sync.sh                              # push code -> /workspace/dino/code (tar|ssh, no rsync on Windows)
bash scripts/run.sh python -m dino.train --config configs/smoke_seg.yaml --set backbone=dinov2_b name=foo
bash scripts/bg.sh myjob python -m dino.train ... # long jobs (nohup); log in /workspace/dino/runs/_jobs/myjob.log
bash scripts/pull.sh [run]                        # metrics/plots/preds -> ./results (no checkpoints)
bash scripts/push_data.sh <local_dir> <name> [--scratch]
bash scripts/run.sh bash tests/smoke/test_pipeline.sh [backbones...]   # end-to-end smoketest
ssh dino-box bash /workspace/dino/code/remote/bootstrap.sh             # re-run after container reset (idempotent)
```
Always `sync.sh` before running edited code remotely.

## Experiments (main workflow: new data -> train -> test benchmark -> report)
- `bash scripts/experiment.sh experiments/<name>.yaml [--data <local_dir>] [--force]` runs `dino.experiment` detached (tmux job `exp_<name>`).
- Pipeline: optional split (`dino/data/split.py`, symlinks, stratified) → `dino/data/validate.py` → per task×backbone subprocess `dino.train` → `dino.evaluate --data_root --split test --out experiments/<name>/eval/<run>.json` → `dino.infer --limit` → baselines → `dino/report.py`.
- Artifacts: `/workspace/dino/experiments/<name>/{experiment.yaml,dataset.json,status.json,results.json,configs/,eval/,experiment.log}`; runs are `runs/<name>__<task>__<backbone>`.
- Report: `/workspace/dino/reports/<name>.html` is self-contained with no JS (JupyterLab's HTML viewer is sandboxed). Open it in JupyterLab at `dino/reports/`. Regenerate with `python -m dino.report <name>`.
- Template: `experiments/_template.yaml`; worked example: `experiments/synthetic_newdata.yaml`.

## Feature cache (dino/features.py)
- On by default whenever the backbone is frozen (`train.cache_features: auto`, `unfreeze_last: 0`). The backbone runs once per (backbone, split, img_size, file signature), and heads train on cached fp16 features: ~8x faster seg training, identical metrics (verified: cached-trained head evaluated on raw images matches within bf16 noise).
- Stored in `/root/scratch/featcache/<backbone>/<hash>/` (ephemeral disk; rebuilt automatically after a reset or any data change). Seg costs ~1.3 MB/image for ViT-B at 336x448; classification is negligible.
- With the cache, train augmentation is only a cached horizontal flip (`cache_flip`), which doubles samples per epoch. Set `train.cache_features: false` for full augmentation. Finetuning (`unfreeze_last > 0`) never caches.
- Cached loaders use `num_workers=0` and `__getitems__` batch fetch; small caches (< `DINO_CACHE_RAM_GB`, default 8) load into RAM, larger ones are memory-mapped.

## Dashboard (local, http://127.0.0.1:8765)
- `.venv/Scripts/python -m uvicorn dashboard.app:app --port 8765` (or preview config `dashboard` in `.claude/launch.json`). Setup: `python -m venv .venv && .venv/Scripts/pip install -r dashboard/requirements.txt`.
- Runs on this PC and pulls data over one persistent paramiko SSH connection (~0.5-1 s per call; a fresh `ssh` handshake costs ~3.3 s at 260 ms RTT). The box needs no open ports.
- Shows GPU util/VRAM/power/temp (via `remote/status.py` → `/opt/venv/bin/amd-smi`), container CPU/RAM, /workspace disk, running processes, job logs, runs with metrics/logs/files, the FoundationPose mask gallery, and `runs/_compare/compare.png`.
- **SSH port forwarding is disabled by the provider** (`AllowTcpForwarding no` in /tmp/amd-oneclick-sshd_config). Don't try to tunnel; don't change that config.
- Remote FastAPI test app (GPU check runs in a subprocess; a resident HIP context spins a CPU core at 100%): `remote/ping_app.py` on 127.0.0.1:8000 (`bash scripts/bg.sh ping uvicorn remote.ping_app:app --host 127.0.0.1 --port 8000`). The dashboard reaches it via `ssh … curl`.
- Only ports 22 (public 36168) and 8888 (provider JupyterLab, root_dir /workspace) listen on the box.

## Box facts
- GPU: AMD Radeon Pro W7900D, 48 GB, gfx1100, ROCm. torch 2.13+rocm comes from `/opt/venv` (Python 3.14), chained into our venv through a `.pth` file. Use `torch.cuda.*` APIs (HIP) with bf16 autocast.
- The torchvision build has **no libpng/libjpeg decoders**, so read images with PIL (`dino.data.datasets.read_rgb`).
- Storage:
  - `/workspace` is 25 GB, persistent NFS with root-squash (tar needs `--no-same-owner`). It holds code/venv/HF cache/data/runs.
  - `/root/scratch` is huge but ephemeral; put large datasets there and symlink them into `data/`.
- Network: pypi, hf-mirror.com, modelscope and codeload.github.com are reachable. huggingface.co, raw.githubusercontent and dl.fbaipublicfiles (torch.hub weights) are blocked.

## Code map
- `dino/backbones.py`: registry (`dinov2_{s,b,l,g}`, `_reg`, `dinov3_{s,b,l}`). Loads with HF `AutoModel`. `$DINO_MODELS/<name>/` overrides the hub, for manual DINOv3 uploads.
- `dino/model.py`: `DinoModel` (backbone + `ClsHead` or `SegHead`), checkpoint save/`load_run`.
- `dino/data/datasets.py`: standard **frames layout** (`rgb/ depth/ masks/ labels.csv cam_K.txt`, root `classes.txt`). ImageFolder is also accepted for classification.
- `dino/data/rgbd.py`: depth/K I/O, `ingest` CLI for raw camera captures, and the FoundationPose `export_frame`.
- `dino/data/synthetic.py`: offline RGB-D smoketest data.
- `dino/train.py`, `evaluate.py`, `infer.py`, `compare.py`: all configured through `configs/*.yaml` plus `--set a.b=v` overrides.
- Runs go to `/workspace/dino/runs/<name>/{config.yaml,log.txt,ckpt.pt,metrics.json}`; comparisons go to `runs/_compare/`.
- `img_size` should be a multiple of 112 so it fits both patch 14 (v2) and patch 16 (v3).
