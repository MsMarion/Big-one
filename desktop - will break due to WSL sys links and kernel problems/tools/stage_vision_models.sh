#!/usr/bin/env bash
# Stage open vision models + offline wheels for the W7900D (no internet there). Run in the pnp_dev container
# (WSL's own python has no venv module):  bash /repo/tools/stage_vision_models.sh   -> /root/vision_staging
set -e
S=/root/vision_staging
mkdir -p $S/models $S/wheels
cd $S
[ -x venv/bin/python ] || python3 -m venv venv
venv/bin/pip install -q --upgrade pip huggingface_hub
venv/bin/python - <<PY
from huggingface_hub import snapshot_download
for repo in ("facebook/dinov2-small", "facebook/sam2.1-hiera-tiny"):
    p = snapshot_download(repo, local_dir=f"$S/models/{repo.split('/')[1]}")
    print("downloaded", repo, "->", p)
PY
# wheels for Python 3.14 on Linux x86_64 (the W7900D's /opt/venv already has torch, torchvision, numpy, pillow, ...)
# --no-deps + an explicit list: the W7900D's /opt/venv already has torch (ROCm), torchvision, numpy, pillow, pyyaml,
# requests, tqdm -- letting pip resolve timm's deps would fetch a ~1 GB CUDA torch for nothing
venv/bin/pip download -q -d wheels --only-binary=:all: --no-deps --python-version 3.14 --implementation cp   --platform manylinux_2_28_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 --platform any   "transformers>=4.56" huggingface_hub hf-xet safetensors tokenizers filelock fsspec packaging regex   typing-extensions scipy opencv-python-headless timm
ls wheels | sed 's/^/  wheel: /'
du -sh models/* wheels
echo STAGE_DONE
