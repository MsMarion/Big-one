#!/usr/bin/env bash
# The Jetson's copy of the repo is a GIT clone (~/pnp, branch `jetson`), the second end of a two-way link
# (docs/learning/22_jetson_d405.md). Run from Git Bash or WSL on the PC, in the repo:
#   bash jetson/sync_to_jetson.sh setup   # once: the repo on the Jetson + the `jetson` remote here
#   bash jetson/sync_to_jetson.sh push    # PC -> Jetson: the committed state of this branch (refused if the Jetson's
#                                         #   tree has uncommitted changes: nothing made there is ever overwritten)
#   bash jetson/sync_to_jetson.sh pull    # Jetson -> PC: commits made there (calibration results), merged here
#   bash jetson/sync_to_jetson.sh commit "msg"   # commit what changed ON the Jetson (e.g. a new calibration file)
#
# A LAN-only remote. Nothing here goes to GitHub or any hosted remote.
# JETSON (default `jetson`, an ssh Host alias for marion@192.168.2.38) and JETSON_DIR (default pnp) override.
set -euo pipefail
J="${JETSON:-jetson}"
D="${JETSON_DIR:-pnp}"
cd "$(dirname "$0")/.."

case "${1:-push}" in
  setup)
    ssh "$J" "test -d $D/.git || git init -q -b jetson $D; git -C $D config receive.denyCurrentBranch updateInstead"
    git remote get-url jetson >/dev/null 2>&1 || git remote add jetson "$J:$D"
    echo "remote jetson -> $(git remote get-url jetson)"
    ;;
  push)
    [ -z "$(git status --porcelain --untracked-files=no)" ] || echo "note: uncommitted changes here are NOT sent (commit first)"
    git push jetson "HEAD:jetson"
    ssh "$J" "cd $D && echo \"jetson now at \$(git log --oneline -1)\" && git status --short | head"
    ;;
  pull)
    git fetch jetson
    if git merge-base --is-ancestor jetson/jetson HEAD; then echo "nothing new on the Jetson"
    else git merge --no-edit jetson/jetson; fi
    ;;
  commit)
    msg="${2:?commit message}"
    ssh "$J" "cd $D && git add -A config/calibration && git status --short && git commit -q -m \"$msg\" && git log --oneline -1"
    ;;
  *)
    echo "usage: $0 setup|push|pull|commit MSG"; exit 2 ;;
esac
