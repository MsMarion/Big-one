#!/usr/bin/env bash
# Copy experiment reports to ./results/reports/ (self-contained HTML, open in any browser).
#   bash scripts/pull_report.sh [name]     (default: all reports + index)
set -euo pipefail
. "$(dirname "$0")/_common.sh"
mkdir -p "$REPO/results/reports"
SEL=${1:+$1.html}
ssh "$HOST" "cd $REMOTE_ROOT/reports && tar czf - ${SEL:-.} index.html" | tar xzf - -C "$REPO/results/reports"
echo "-> $REPO/results/reports/${SEL:-}"
