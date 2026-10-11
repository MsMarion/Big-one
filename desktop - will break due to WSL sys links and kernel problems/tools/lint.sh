#!/usr/bin/env bash
# pyflakes over our own code (not the vendored packages). Exit 1 on any finding.   bash /repo/tools/lint.sh [files]
cd "$(dirname "$0")/.." || exit 2
files=("$@")
[ ${#files[@]} -eq 0 ] && files=(pnp_core sim tools setup tests ros2_ws/src/pnp_perception/pnp_perception ros2_ws/src/pnp_manipulation/pnp_manipulation ros2_ws/src/pnp_bringup/launch)
/usr/bin/python3 -m pyflakes "${files[@]}"
