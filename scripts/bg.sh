#!/usr/bin/env bash
# Launch a long job detached:  scripts/bg.sh <job_name> python -m dino.train --config configs/x.yaml
# Follow:  scripts/run.sh tail -f /workspace/dino/runs/_jobs/<job_name>.log
set -euo pipefail
. "$(dirname "$0")/_common.sh"
ssh "$HOST" "bash $REMOTE_CODE/remote/bg.sh $*"
