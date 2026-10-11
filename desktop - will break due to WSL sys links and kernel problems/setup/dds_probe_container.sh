#!/usr/bin/env bash
# Link test, container side: a throwaway container with the zenoh bridge + the probe "pong".
# Then on Windows: start setup\zenoh_windows.ps1, and run the probe "ping" (see setup/dds_probe.py).
# Stop with:  bash ~/pnp_repo/docker/pnp_docker.sh rm -f pnp_probe
set -euo pipefail
D=(/usr/bin/docker -H unix:///var/run/docker-pnp.sock)
"${D[@]}" rm -f pnp_probe >/dev/null 2>&1 || true
"${D[@]}" run -d --name pnp_probe -p 127.0.0.1:7447:7447 -v "$HOME/pnp_repo/":/repo pnp:jazzy \
  bash -c 'bash /repo/setup/zenoh_container.sh > /tmp/zenoh.log 2>&1 &
           source /repo/setup/ros_env.sh && python3 -u /repo/setup/dds_probe.py pong' >/dev/null
sleep 4
"${D[@]}" logs pnp_probe
