#!/usr/bin/env bash
# Container side of the Windows <-> container ROS 2 link (started by docker/run_dev.sh).
# Runs zenoh-bridge-ros2dds as a zenoh ROUTER listening on TCP 7447 (published to the WSL host's
# 127.0.0.1:7447, which Windows reaches through WSL's shared loopback). It discovers the ROS nodes in
# this container over normal DDS and forwards their topics/services/actions through zenoh.
#
# Self-healing supervisor: when the Windows bridge disconnects and later rejoins, this router does NOT
# re-create the routes for the topics it publishes into the container (zenoh-plugin-ros2dds 1.10.1,
# observed 2026-10-11: /clock and /isaac_joint_states silently stop). A FRESH router does it correctly,
# and the Windows bridge reconnects to a fresh router within ~5 s. So: whenever the remote bridge
# leaves, restart the router.
source /repo/setup/ros_env.sh
export CYCLONEDDS_URI="file:///repo/config/cyclonedds_bridge.xml"   # big socket buffers for images
LOG=/tmp/zenoh.log
while true; do
  # Optional: also connect OUT to a Jetson streaming the real camera (config/remote_sensors.yaml).
  CONNECT=$(python3 -c "
import yaml; j = yaml.safe_load(open('/repo/config/remote_sensors.yaml'))['jetson']
print('--connect ' + j['endpoint'] if j.get('enabled') else '')" 2>/dev/null)
  [ -n "$CONNECT" ] && echo "[zenoh_container] $(date -u +%T) also connecting to the Jetson: $CONNECT" >> "$LOG"
  # shellcheck disable=SC2086  (CONNECT is empty or two words)
  zenoh-bridge-ros2dds router --listen tcp/0.0.0.0:7447 $CONNECT --domain "${ROS_DOMAIN_ID}" --no-multicast-scouting \
    -c /repo/config/zenoh_bridge.json5 >> "$LOG" 2>&1 &
  pid=$!
  # Block until the remote (Windows) bridge leaves, or the router itself exits.
  tail -n0 -F "$LOG" --pid="$pid" 2>/dev/null | grep -m1 -q "Remote ROS 2 bridge left"
  echo "[zenoh_container] $(date -u +%T) remote bridge left (or router exited) -> restarting router" >> "$LOG"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  sleep 1
done
