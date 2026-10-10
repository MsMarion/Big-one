#!/usr/bin/env bash
# Run a command on the box inside the project env:  scripts/run.sh python -m dino.train --config configs/x.yaml
set -euo pipefail
. "$(dirname "$0")/_common.sh"
ssh "$HOST" "source $REMOTE_CODE/remote/env.sh && cd $REMOTE_CODE && $*"
