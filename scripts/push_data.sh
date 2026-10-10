#!/usr/bin/env bash
# Upload a local dataset folder:  scripts/push_data.sh <local_dir> <name> [--scratch]
# --scratch puts it on the big ephemeral disk (/root/scratch) and symlinks it into /workspace/dino/data.
set -euo pipefail
. "$(dirname "$0")/_common.sh"
SRC=$1; NAME=$2; MODE=${3:-}
if [ "$MODE" = "--scratch" ]; then
  DEST=/root/scratch/data/$NAME
  ssh "$HOST" "mkdir -p $DEST $REMOTE_ROOT/data && ln -sfn $DEST $REMOTE_ROOT/data/$NAME"
else
  DEST=$REMOTE_ROOT/data/$NAME
  ssh "$HOST" "mkdir -p $DEST"
fi
tar czf - -C "$SRC" . | ssh "$HOST" "tar xzf - --no-same-owner -C $DEST"
echo "uploaded $SRC -> $HOST:$DEST"
