#!/usr/bin/env bash
# Record a whole session to an MCAP bag: both cameras, /tf, the joints, what perception and the pick thought
# (the topic list: config/recording.yaml session_topics). The bag is listed in the trial log (pnp_core/trials.py
# artifacts) so it can be found next to the runs and attempts it belongs to.
#
#   bash /repo/tools/record_session.sh [name] [seconds]      (in the pnp container; seconds 0 = until Ctrl+C)
#   from WSL:  pnp run "bash /repo/tools/record_session.sh demo1 60"
#
# Replay later:  ros2 bag play /data/bags/<name>      (e.g. into perception alone, to test new code on old data)
# Sizes: ~53 MB/s of camera images in the sim -> about 3 GB a minute.
set -euo pipefail
NAME=${1:-session_$(date +%Y%m%d_%H%M%S)}
SECS=${2:-0}
OUT=/data/bags/$NAME
mkdir -p /data/bags
mapfile -t TOPICS < <(python3 -c "
import yaml
for t in yaml.safe_load(open('/repo/config/recording.yaml'))['session_topics']: print(t)")
echo "[record_session] $OUT: ${#TOPICS[@]} topics, $([ "$SECS" -gt 0 ] && echo "$SECS s" || echo 'until Ctrl+C')"
CMD=(ros2 bag record -s mcap --include-hidden-topics --qos-profile-overrides-path /repo/config/recording_qos.yaml
     -o "$OUT" --topics "${TOPICS[@]}")
if [ "$SECS" -gt 0 ]; then timeout --signal=INT "$SECS" "${CMD[@]}" || true; else "${CMD[@]}" || true; fi
ros2 bag info "$OUT" | grep -E "Duration|Messages|Topic:" || true
python3 -c "
import sys; sys.path.insert(0, '/repo')
from pnp_core import trials
print('[record_session] artifact', trials.artifact('session_bag', '$OUT', note='tools/record_session.sh $NAME'))"
