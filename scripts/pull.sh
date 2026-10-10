#!/usr/bin/env bash
# Fetch run outputs (metrics, plots, small preds; no checkpoints) into ./results/
#   scripts/pull.sh            # all runs
#   scripts/pull.sh <run_name> # one run
set -euo pipefail
. "$(dirname "$0")/_common.sh"
SEL=${1:-.}
mkdir -p "$REPO/results"
ssh "$HOST" "cd $REMOTE_ROOT/runs && tar czf - --exclude='*.pt' --exclude='*.safetensors' --exclude=_jobs $SEL" \
  | tar xzf - -C "$REPO/results"
echo "pulled -> $REPO/results"
