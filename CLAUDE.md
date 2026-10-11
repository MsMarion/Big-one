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

## objsearch (local PC: RGB-D object -> identity -> 3D model -> Blender -> USD)
Runs on this PC, not on the box: the box can't reach the LLM APIs or the model sites. Plan: `~/.claude/plans/i-need-to-build-sunny-map.md` (phases 1-7). Phase 1 is done: the SQLite DB, tracking, fakes, dashboard and tests.
- State lives in `objsearch_data/objsearch.db` (gitignored; SQLite WAL, `PRAGMA user_version` migrations in `objsearch/db.py`). Files sit under `objsearch_data/{photos,runs,assets}`, and DB rows store paths relative to it. Env `OBJSEARCH_DATA` overrides the location (tests use it).
- Catalog: `catalog/seed.yaml` holds about 58 seed objects. `python -m objsearch db import|export|stats`. Export round-trips the DB, including photos, assets and evidence.
- Every pipeline action runs inside `track.run()` / nested `track.step()` (`objsearch/track.py`). These write `steps`, `llm_calls`, `tool_calls`, `http_requests` and `quota_usage` rows, plus `runs/<id>/run.log.jsonl`. Secrets are redacted.
- LLM: one interface, `objsearch/llm/base.py`. Tool schemas live in `llm/tools.py` and must stay in the Gemini/Claude-portable JSON-schema subset (a test enforces this). `FakeLLM` plays scripted turns. Default provider: Gemini free tier (phase 4).
- `python -m objsearch demo [--fail-at identify|acquire|convert]` runs a fake end-to-end pipeline. `demo --clean` removes its runs and `demo-*` objects.
- Reference images: `python -m objsearch refimages [--ids ..] [--query ..] [--force]` searches Openverse with no key (anonymous limit 200 searches/day). Images arrive through its thumbnail proxy, because Flickr and Wikimedia refuse script downloads. LEGO parts with an `ldraw_id` use the exact LDraw.org render (CC BY 4.0) instead. License and attribution are stored in `ref_images` and must be shown wherever an image appears.
- Gripper fit: items are judged against the SO-ARM101 gripper (`objsearch/grasp.py`, limits in `configs/objsearch.yaml -> gripper`). An item passes when it is >= 10 mm tall and 10 to (opening - clearance) mm across the jaws. `dims_mm` is [L, W, H] as the item sits; `grasp_pose: flat` overrides that. `max_open_mm: 55` is an estimate until measured. The dashboard and the Check page default to `grip=ok` (good + borderline). Check the whole catalog with `python -m objsearch grasp`.
- Identify from photos: use `/objects/identify` (drop 1-6 photos of ONE object, or use the phone camera button) or `python -m objsearch identify <files|folder> [--hint ..] [--dims L W H]`. The pipeline in `objsearch/identify.py` runs intake, then identify (the Gemini agent with catalog_lookup, Brickognize, grounded web_search, web_fetch, and a final `submit_result` call), then record (matches the catalog or creates a `found` object), then the user confirms or corrects it. With no `GEMINI_API_KEY` it runs keyless and only identifies LEGO, via Brickognize. Gemini backend: `objsearch/llm/gemini_backend.py`. `web_fetch` follows redirects by hand and refuses private hosts.
- 3D models: use `python -m objsearch model add <object_id> <url|file> --license .. --source-url ..` or "Add a 3D model" on the object page. The pipeline in `objsearch/models.py` runs download (quarantine, allowlist, no `.blend`), then convert (STEP→GLB via cascadio; PCB designs via `objsearch/kicad.py`), then normalize, then register. Normalize (`objsearch/blender/normalize.py`) runs in `.venv-bpy`: it fixes units against `dims_mm`, puts the base on z=0, and writes asset.usd (metres, Z-up), model.glb, mesh/mesh.obj, a Cycles thumb.png and scene.blend under `objsearch_data/model_cache/<build key>/`. Size checks use the footprint (two largest sides, ±15%); a height difference only gets a note, since pin headers vary. Meshes under 100 triangles get flagged as boxes.
- Auto-acquire (`objsearch/acquire.py`, `python -m objsearch acquire <ids> [--no-agent] [--force]`) finds the closest replica of the exact item. Every candidate gets a tier, and the best tier wins:
  1. exact board: the product's own `.kicad_pcb`/`.brd` from GitHub, built with KiCad; LEGO uses the LDraw part
  2. detailed STEP: >= 2000 triangles or >= 12 parts
  3. KiCad replica (`catalog/boards/<id>.yaml`)
  4. simple STEP, kept only as a fallback
  5. photo box
  - Tier 1-2 hits build at once. Otherwise the best of tiers 3-5 is built right away, then the research agent hunts for tier 1-2 and replaces the model only with a better tier.
  - Learned hints store their tier. A STEP hint with no recorded tier counts as tier 4.
  - One failing source never ends the ladder.
  - Before an item gets a model it gets a size (`physical.ensure_dims`).
- GitHub (`sources/github.py`) uses the `gh` login. Code search allows 10 requests/min. Queries are cleaned with `clean_query`, results are cached per process, and a rate limit waits for the reset once. Unit tests set `OBJSEARCH_NO_GITHUB=1`.
- KiCad replicas (`objsearch/boardgen.py`, spec `catalog/boards/<id>.yaml`) give every value its source. A spec can contain:
  - board outline, thickness, mask, silk and finish
  - unplated or plated holes
  - a header with `state: loose|soldered`
  - front parts (library or fetched footprints/models, tinted with `parts.tint_step`)
  - `back_parts` placed from a perspective-corrected photo (`back_calibration`) and flipped with KiCad's own pcbnew (`objsearch/kicad_flip.py`, run with KiCad's Python)
  - `back_silk`
  - `overlay`: face groups of a real STEP matched by size and aligned by the STEP's own PCB
  - `extras`: reconstructed thin boxes
  `build_model` merges these into one GLB. For side-by-side checks, `normalize --views top,bottom` renders orthographic views.
- Recolour (`models.add_model(..., recolor=)`, stored on the hint) matches parts by measured size, including material slots of merged meshes. It refuses ambiguous names.
- The sim layer uses convex hulls for up to 32 meshes. Above that it uses one box collider.

- mesh/mesh.obj is written in the USD's frame (Z-up, metres; `obj_up: "Z"` in result.json). Builds older than PIPELINE_VERSION 2026-10-11b had Y-up OBJs, and `export` rebuilds those automatically.
- Sim USD (`objsearch/sim.py` + `blender/usd_layer.py`): `assets/<id>/sim.usda` references the shared geometry and adds a rigid body, mass, convex-hull colliders, a friction/restitution physics material, `UsdSemantics` labels (class = object id, plus category) and the legacy Isaac `semantic:Semantics:*` attributes. It's rebuilt after every model build and after docs research (about 0.5 s). Run `python -m objsearch sim [ids]` to rebuild by hand. Physics come from `physical.physics_for`: researched values beat category estimates.
- Docs (`objsearch/docs.py`, `python -m objsearch docs <ids> [--force]`, runs after acquire when `docs.auto`): a Gemini agent with web_search (soft limit of 3), web_fetch and `grab_datasheet`. It saves the PDF (guarded download, `%PDF` check) to `<data>/docs/<id>/` as `evidence` with kind `datasheet` and a `local_path`. It writes `objects.physics` (mass, materials, friction, restitution, fragile parts) and clamps implausible values. LEGO is skipped. It takes 40-75 s. The object page has a "Physical & simulation" card with a "Research datasheet" button.
- Export for DINOv3 / FoundationPose / Isaac: `python -m objsearch export <dir> [--ids ..] [--owned] [--real-only]`. It writes `classes.txt` (class id = line index), BOP-style `models_info.json` (diameter/bbox in mm, mass, quality real|exact|placeholder, license), and per object `mesh/` (FoundationPose OBJ + texture), `usd/sim.usda` (self-contained via `UsdUtils.LocalizeAsset`), model.glb, thumb.png and info.json.
- KiCad (`objsearch/kicad.py`, KiCad 10 at `C:/Program Files/KiCad/10.0`) turns any board (.kicad_pcb, Eagle .brd, Altium… or a zip) into STEP: `kicad-cli pcb import`, then sanitize (undefined Eagle layers), then attach the official KiCad 3D models by fitting library pads to board pads (`EAGLE_TO_KICAD` map; `_CONV` sign convention was verified by render), then `pcb export step`. Missing models come from the kicad-packages3D GitLab and are cached. KiCad has no USB-B 3D model. `kicad.find(q)` searches the ~7k installed models.
- Parts library (`objsearch/parts.py`, table `parts3d`, page `/objects/parts`, CLI `python -m objsearch parts list|index|add|rebuild`). Every board build resolves each part's 3D model in this order: library real, KiCad installed, KiCad online, library approximate, generate an approximation. Approximations are gmsh boxes from the footprint's F.Fab outline (KiCad parts) or the board's own outline (unmapped parts), with height from `HEIGHT_HINTS_MM`. Each one is flagged on the model (`models.part_flags`), and the inventory card shows "3D: n approx". Unmapped board footprints are first matched to KiCad footprints by name-free pad geometry (`kicad.suggest_footprint`, needs `parts index`, ~2 min once); a good match is learned in parts3d. A real STEP added for a part is auto-aligned (`parts.auto_align` + `orient_by_pads`: the leads must land on the pads, which resolves front/back and up/down) and triggers a rebuild of the objects that used it. The USB-B (FreeCAD-library, CC-BY-3.0) is the first real library part. Approximations and gmsh-written parts carry no colors.
- Dashboard: `/objects`, `/objects/check` (go through the catalog, "do I have it?"), `/objects/<id>`, `/objects/runs` in the local dashboard (`dashboard/objects.py`). Preview config `dashboard-alt` serves it on port 8766.
- Tests: `bash scripts/test_objsearch.sh [unit|bpy|replay|live|bench]`. Unit is the default and uses fakes only. `live` needs `--max-requests`.
- Deps: local `.venv` from `dashboard/requirements.txt` (includes `objsearch/requirements.txt`) and `objsearch/requirements-dev.txt`. Blender goes in `.venv-bpy` (pip `bpy==5.2.*`, Python 3.13), arriving in phase 2.
