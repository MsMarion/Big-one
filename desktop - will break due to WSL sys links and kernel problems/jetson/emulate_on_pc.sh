#!/usr/bin/env bash
# Pretend to be the Jetson, on this PC, to test the whole network camera path WITHOUT hardware:
#
#   recorded camera bag --(ros2 bag play)--> raw topics --(image_transport: JPEG / PNG, like the Jetson)-->
#   compressed topics --(zenoh bridge with jetson/zenoh_jetson.json5)--> TCP --> PC container router
#   --> bringup.launch.py camera:=jetson decompresses --> detector (same as with the real camera)
#
# Run inside WSL (Ubuntu-24.04):
#   bash ~/pnp_repo/jetson/emulate_on_pc.sh start [bag]   # bag default: /data/bags/sim_camera_ref
#                                                         #   (record one: tools/record_camera.sh sim_camera_ref)
#   bash ~/pnp_repo/jetson/emulate_on_pc.sh stop
# `start` prints the endpoint to put into config/remote_sensors.yaml.
#
# The emulator uses ROS_DOMAIN_ID 43, the PC 42: otherwise DDS discovery between the two containers (same Docker
# network) would deliver the RAW bag topics directly and the test would prove nothing about the network path.
set -euo pipefail
DOCKER=(/usr/bin/docker -H unix:///var/run/docker-pnp.sock)
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
NAME=pnp_jetson_emu

case "${1:-start}" in
  stop)
    "${DOCKER[@]}" rm -f "$NAME" >/dev/null 2>&1 && echo "emulator stopped" ;;
  start)
    BAG=${2:-/data/bags/sim_camera_ref}
    "${DOCKER[@]}" rm -f "$NAME" >/dev/null 2>&1 || true
    "${DOCKER[@]}" run -d --name "$NAME" --hostname jetson-emu -e ROS_DOMAIN_ID=43 \
      -v "$REPO":/repo:ro -v "$HOME/pnp_data":/data pnp:jazzy bash -c "
        source /opt/ros/jazzy/setup.bash
        sed 's/domain: 42/domain: 43/' /repo/jetson/zenoh_jetson.json5 > /tmp/zenoh_emu.json5
        zenoh-bridge-ros2dds -c /tmp/zenoh_emu.json5 > /tmp/zenoh.log 2>&1 &
        T=/camera/camera
        ros2 run image_transport republish --ros-args -p in_transport:=raw -p out_transport:=compressed \
          -r in:=\$T/color/image_raw -r out/compressed:=\$T/color/image_raw/compressed > /tmp/rep_color.log 2>&1 &
        ros2 run image_transport republish --ros-args -p in_transport:=raw -p out_transport:=compressedDepth \
          -r in:=\$T/aligned_depth_to_color/image_raw \
          -r out/compressedDepth:=\$T/aligned_depth_to_color/image_raw/compressedDepth > /tmp/rep_depth.log 2>&1 &
        exec ros2 bag play --loop $BAG > /tmp/play.log 2>&1" >/dev/null
    sleep 2
    IP=$("${DOCKER[@]}" inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$NAME")
    echo "emulated Jetson running: endpoint tcp/$IP:7447 (put it into config/remote_sensors.yaml, enabled: true,"
    echo "then: bash /repo/setup/restart_zenoh_container.sh in the pnp container)" ;;
  *)
    echo "usage: $0 start [bag] | stop"; exit 2 ;;
esac
