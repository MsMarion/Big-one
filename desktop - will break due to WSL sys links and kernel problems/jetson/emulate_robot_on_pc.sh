#!/usr/bin/env bash
# Pretend to be the Jetson in the ROBOT role, on this PC: the robot side of the stack (mock servo bus, no camera)
# in a second container, linked to the PC's container ONLY through zenoh with the Jetson's allow-list
# (jetson/zenoh_robot.json5). Tests the split (MoveIt / controllers / services over the link) without hardware.
#
# Run inside WSL (Ubuntu-24.04), with the pnp container up (docker/run_dev.sh) and its workspace built:
#   bash ~/pnp_repo/jetson/emulate_robot_on_pc.sh start      # prints the endpoint for config/remote_sensors.yaml
#   bash ~/pnp_repo/jetson/emulate_robot_on_pc.sh stop
# Then in the pnp container: restart its router, and start the brain:
#   ros2 launch pnp_bringup bringup.launch.py role:=brain sim:=false camera:=none manipulation:=true \
#     scene:=config/scenes/so101_workcell.yaml
# ROS domain 43 here, 42 on the PC: plain DDS between the containers can't shortcut the zenoh path.
set -euo pipefail
DOCKER=(/usr/bin/docker -H unix:///var/run/docker-pnp.sock)
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
NAME=pnp_jetson_robot_emu

case "${1:-start}" in
  stop)
    "${DOCKER[@]}" rm -f "$NAME" >/dev/null 2>&1 && echo "robot emulator stopped" ;;
  start)
    "${DOCKER[@]}" rm -f "$NAME" >/dev/null 2>&1 || true
    "${DOCKER[@]}" run -d --name "$NAME" --hostname jetson-robot-emu --init -e ROS_DOMAIN_ID=43 \
      -e PNP_VENV_PYTHON=/opt/pnp_venv/bin/python3 \
      -v "$REPO":/repo:ro -v "$HOME/pnp_ws":/ws:ro -v "$HOME/pnp_data":/data pnp:jazzy bash -c "
        source /repo/setup/ros_env.sh
        sed 's/domain: 42/domain: 43/' /repo/jetson/zenoh_robot.json5 > /tmp/zenoh_robot.json5
        zenoh-bridge-ros2dds -c /tmp/zenoh_robot.json5 > /tmp/zenoh.log 2>&1 &
        exec ros2 launch pnp_bringup bringup.launch.py role:=robot sim:=false hardware:=real hw_mock:=true \
          camera:=none manipulation:=true scene:=config/scenes/so101_workcell.yaml use_rviz:=false \
          > /tmp/robot.log 2>&1" >/dev/null
    sleep 2
    IP=$("${DOCKER[@]}" inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$NAME")
    echo "emulated Jetson (robot role) running: endpoint tcp/$IP:7447"
    echo "logs: docker exec $NAME tail -f /tmp/robot.log" ;;
  *)
    echo "usage: $0 start | stop"; exit 2 ;;
esac
