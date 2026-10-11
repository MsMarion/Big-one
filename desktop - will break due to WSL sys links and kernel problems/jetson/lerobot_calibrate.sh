#!/usr/bin/env bash
# Fresh LeRobot calibration of the SO-101 follower on the Jetson HOST (interactive: you move the arm), then the result
# goes where our stack reads it. docs/learning/22_jetson_d405.md, step 3.
#   bash ~/pnp/jetson/lerobot_calibrate.sh            (any shell; no conda env needed)
#
# Why the wrapper: the Jetson's LeRobot (0.3.3) lives in ~/.local for the system Python 3.10. There, deepdiff imports
# Ubuntu's pandas (built for numpy 1.x) next to numpy 2.2 and crashes ("numpy.dtype size changed"). Calibration
# doesn't use pandas, and deepdiff treats it as optional, so it's hidden (ImportError) instead of changing any install.
set -euo pipefail
ID="${ROBOT_ID:-desk}"
PORT="${PORT:-/dev/ttyACM0}"
OUT="$(cd "$(dirname "$0")/.." && pwd)/config/calibration/so101_desk/so101_follower.json"
SRC="$HOME/.cache/huggingface/lerobot/calibration/robots/so101_follower/$ID.json"

echo "LeRobot calibration of the follower on $PORT (id $ID). The torque stays OFF: support the arm."
/usr/bin/python3 -c '
import sys
sys.modules["pandas"] = None          # see the header: optional for deepdiff, broken against numpy 2 here
sys.modules["polars"] = None
from lerobot.calibrate import main
main()' --robot.type=so101_follower --robot.port="$PORT" --robot.id="$ID"

cp "$SRC" "$OUT"
echo "copied $SRC -> $OUT"
echo "next (from the PC): bash jetson/sync_to_jetson.sh commit \"SO-101 servo calibration\" && bash jetson/sync_to_jetson.sh pull"
