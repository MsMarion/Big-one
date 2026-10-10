# Source this before any remote command:  source /workspace/dino/code/remote/env.sh
# Non-interactive SSH does not read ~/.bashrc, so the egress proxy must be set here.

_PROXY=http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128
export http_proxy=$_PROXY https_proxy=$_PROXY HTTP_PROXY=$_PROXY HTTPS_PROXY=$_PROXY
export no_proxy=127.0.0.1,localhost,.svc,.svc.cluster.local
export NO_PROXY=$no_proxy

export DINO_ROOT=/workspace/dino
export DINO_CODE=$DINO_ROOT/code
export DINO_DATA=$DINO_ROOT/data
export DINO_RUNS=$DINO_ROOT/runs
export DINO_MODELS=$DINO_ROOT/models
export DINO_SCRATCH=/root/scratch          # big but ephemeral (container overlay)

# huggingface.co is blocked by the proxy; hf-mirror.com works.
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$DINO_ROOT/cache/hf
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1   # xet CAS storage bypasses the mirror (401); force plain HTTP downloads
export TORCH_HOME=$DINO_ROOT/cache/torch
export PIP_CACHE_DIR=$DINO_SCRATCH/pip-cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [ -f "$DINO_ROOT/venv/bin/activate" ]; then
  . "$DINO_ROOT/venv/bin/activate"
else
  . /opt/venv/bin/activate
fi
export PYTHONPATH=$DINO_CODE${PYTHONPATH:+:$PYTHONPATH}
