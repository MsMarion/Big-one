#!/usr/bin/env bash
# Push local code to the box (no rsync on Windows, so tar | ssh). Remote-only dirs are never touched.
set -euo pipefail
. "$(dirname "$0")/_common.sh"
cd "$REPO"
tar czf - --exclude=.git --exclude=./.venv --exclude=./.claude --exclude=results --exclude=__pycache__ --exclude='*.pyc' \
  --exclude=./data --exclude=./runs . \
  | ssh "$HOST" "mkdir -p $REMOTE_CODE && tar xzf - --no-same-owner -m -C $REMOTE_CODE && find $REMOTE_CODE -name '*.sh' -exec chmod +x {} +"
echo "synced -> $HOST:$REMOTE_CODE"
