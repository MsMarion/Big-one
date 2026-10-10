#!/usr/bin/env bash
# Idempotent server setup. Safe to re-run after a container restart.
#   ssh dino-box bash /workspace/dino/code/remote/bootstrap.sh
set -euo pipefail
. "$(dirname "$0")/env.sh"

mkdir -p "$DINO_DATA" "$DINO_RUNS" "$DINO_MODELS" "$HF_HOME" "$TORCH_HOME" "$DINO_SCRATCH" "$PIP_CACHE_DIR"

if [ ! -x "$DINO_ROOT/venv/bin/python" ]; then
  echo "[bootstrap] creating venv on top of /opt/venv (reuses ROCm torch)"
  /opt/venv/bin/python -m venv --system-site-packages "$DINO_ROOT/venv"
fi
. "$DINO_ROOT/venv/bin/activate"
# /opt/venv is itself a venv, so --system-site-packages alone doesn't see its ROCm torch; chain it via .pth
OPT_SP=$(/opt/venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
MY_SP=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "$OPT_SP" > "$MY_SP/_opt_venv.pth"

echo "[bootstrap] pip install"
python -m pip install -q --upgrade pip
python -m pip install -q -r "$DINO_CODE/remote/requirements.txt"

# No sudo / apt here by policy: system packages are not touched. bg.sh uses tmux only if already present, else nohup.

echo "[bootstrap] env check"
python -m tests.smoke.test_env
echo "[bootstrap] done"
