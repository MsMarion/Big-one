#!/usr/bin/env bash
# objsearch test tiers, run locally (not on the box):
#   bash scripts/test_objsearch.sh            # unit: fakes only, seconds
#   bash scripts/test_objsearch.sh bpy        # real Blender (.venv-bpy), fixture meshes
#   bash scripts/test_objsearch.sh replay     # recorded LLM/HTTP cassettes
#   bash scripts/test_objsearch.sh live --max-requests 20 [--budget-usd 0.50]   # real APIs, opt-in
#   bash scripts/test_objsearch.sh bench      # benchmark sets -> evals table
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python; [ -x "$PY" ] || PY=.venv/bin/python
tier="${1:-unit}"; shift || true
case "$tier" in
  unit)                  exec "$PY" -m pytest tests/objsearch -q "$@" ;;
  bpy|replay|live|bench) exec "$PY" -m pytest tests/objsearch -q -m "$tier" "$@" ;;
  *) echo "unknown tier: $tier (unit|bpy|replay|live|bench)"; exit 2 ;;
esac
