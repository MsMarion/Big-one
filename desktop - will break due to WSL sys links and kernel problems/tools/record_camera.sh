#!/usr/bin/env bash
# Record the camera topics (raw color + aligned depth + camera_info) to an MCAP bag.
#   bash /repo/tools/record_camera.sh [name] [seconds]          (in the pnp container)
# Uses: the "real-val" set (record the REAL D435i, label later), replaying into perception offline
# (ros2 bag play), and jetson/emulate_on_pc.sh (pretend-Jetson test without hardware).
set -euo pipefail
NAME=${1:-camera_$(date +%Y%m%d_%H%M%S)}
SECS=${2:-15}
OUT=/data/bags/$NAME
mkdir -p /data/bags
T=/camera/camera
timeout --signal=INT "$SECS" ros2 bag record -s mcap -o "$OUT" --topics \
  $T/color/image_raw $T/color/camera_info $T/aligned_depth_to_color/image_raw $T/aligned_depth_to_color/camera_info \
  || true
ros2 bag info "$OUT" | grep -E "Duration|Messages|Topic:"
