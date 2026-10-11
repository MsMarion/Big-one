#!/usr/bin/env bash
# Look inside the running project from a WSL terminal (e.g. VS Code connected to WSL). docs/learning/28_how_it_runs.md
#
#   bash setup/pnp.sh status              what runs: Isaac (Windows), the bridges, the container, the ROS nodes
#   bash setup/pnp.sh shell               a bash shell INSIDE the container (ROS sourced: ros2 topic list, ...)
#   bash setup/pnp.sh run "<command>"     one command inside the container, e.g. "ros2 topic hz /joint_states"
#   bash setup/pnp.sh logs [ros|sim|zenoh|zenoh-win|camera]   follow a log live (Ctrl+C stops watching, nothing else)
#   bash setup/pnp.sh py <script> [args]  a Python script inside the container with ROS sourced (data/rospy.sh)
#   bash setup/pnp.sh docker <args>       docker on the PROJECT's engine (plain `docker` here is a different one)
#
# Tip: alias pnp='bash ~/pnp_repo/setup/pnp.sh'   (then: pnp status, pnp logs ros, ...)
# Starting / restarting things is setup\stack.ps1 (Windows PowerShell); this script only looks and runs commands.
set -uo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
DOCKER=(docker -H unix:///var/run/docker-pnp.sock)
# docker/run_dev.sh: starts the container if needed, then `docker exec pnp_dev bash -ic "<cmd>"` (ROS sourced)
in_container() { bash "$REPO/docker/run_dev.sh" "$@"; }
newest_sim_log() { ls -t "$REPO"/data/logs/run_sim_*.log 2>/dev/null | head -1; }

what="${1:-status}"; shift || true
case "$what" in
  status)
    echo "== Windows (seen from WSL) =="
    kit=$(tasklist.exe /FI "IMAGENAME eq kit.exe" /NH 2>/dev/null | grep -c kit.exe)
    zb=$(tasklist.exe /FI "IMAGENAME eq zenoh-bridge-ros2dds.exe" /NH 2>/dev/null | grep -c zenoh)
    echo "Isaac Sim (kit.exe):    $([ "$kit" -gt 0 ] && echo "running ($kit processes)" || echo "NOT running")"
    echo "zenoh bridge (Windows): $([ "$zb" -gt 0 ] && echo running || echo "NOT running")"
    echo "newest sim log:         $(newest_sim_log)"
    echo "== container =="
    "${DOCKER[@]}" ps --format 'container {{.Names}}: {{.Status}} ({{.Image}})'
    in_container "echo; echo 'ros2 launch:'; pgrep -fa 'ros2 launch pnp_bringup [b]ringup' | cut -c1-160 || echo '  NOT running'; echo; echo 'ROS nodes:'; ros2 node list 2>/dev/null | grep -v -E 'transform_listener|_private_|^/moveit_[0-9]' | sed 's/^/  /'"
    ;;
  shell) in_container ;;
  run) in_container "$*" ;;
  py) in_container "bash /repo/data/rospy.sh /repo/$*" ;;
  docker) "${DOCKER[@]}" "$@" ;;
  logs)
    case "${1:-ros}" in
      ros) tail -n 40 -F "$HOME/pnp_data/logs_bringup.log" ;;          # every ROS node's output (= /data in the container)
      camera) tail -n 40 -F "$HOME/pnp_data/camera_keepalive.log" ;;
      zenoh) in_container "tail -n 40 -F /tmp/zenoh.log" ;;              # the container's zenoh router
      zenoh-win) tail -n 40 -F "$REPO/data/logs/zenoh_windows.log" ;;
      sim) f=$(newest_sim_log); echo "following $f"; tail -n 40 -F "$f" ;;
      *) echo "logs: ros | sim | zenoh | zenoh-win | camera" ;;
    esac
    ;;
  *) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
