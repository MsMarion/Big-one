#!/usr/bin/env bash
# Remote side of scripts/bg.sh. Runs "$@" detached in the project env, logging to runs/_jobs/<name>.log
set -euo pipefail
. "$(dirname "$0")/env.sh"
NAME=$1; shift
mkdir -p "$DINO_RUNS/_jobs"
LOG=$DINO_RUNS/_jobs/$NAME.log
CMD=$(printf '%q ' "$@")
cd "$DINO_CODE"
if command -v tmux >/dev/null; then
  tmux new-session -d -s "$NAME" "bash -c 'source remote/env.sh && $CMD 2>&1 | tee $LOG'"
  echo "tmux session: $NAME   (attach: tmux attach -t $NAME)"
else
  nohup bash -c "source remote/env.sh && $CMD" > "$LOG" 2>&1 < /dev/null &
  echo "pid: $!"
fi
echo "log: $LOG"
