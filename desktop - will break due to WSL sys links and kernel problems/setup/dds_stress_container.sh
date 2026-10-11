#!/usr/bin/env bash
# Link stress test, container side: start N extra ROS processes (one DDS participant each) in the
# probe container. Then on Windows check they are all visible through the bridge:
#   C:\isaac-sim\python.bat setup\dds_probe.py list 8 /pnp_stress
set -euo pipefail
N="${1:-12}"
/usr/bin/docker -H unix:///var/run/docker-pnp.sock exec -d pnp_probe bash -c \
  "source /repo/setup/ros_env.sh; for i in \$(seq 1 $N); do ros2 topic pub -r 1 /pnp_stress/t\$i std_msgs/msg/Empty '{}' >/dev/null 2>&1 & done; wait"
echo "started $N publishers"
