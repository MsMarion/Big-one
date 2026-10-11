#!/usr/bin/env bash
# Open a shell in the pnp dev container (starts it if needed; extra calls attach new shells).
# Run inside WSL Ubuntu-24.04:   bash ~/pnp_repo/docker/run_dev.sh [command...]
#
#   bridge network (Docker default) -> DDS discovery inside the container works reliably.
#                                      (Host networking on mirrored WSL does NOT: see docs/learning/00_setup.md)
#   -p 127.0.0.1:7447:7447 -> the zenoh bridge's TCP port; Windows reaches it via WSL's shared loopback.
#                             Bound to loopback only, so it is not exposed on your LAN.
#   --init                 -> a real init as PID 1 that reaps exited processes (no zombie pile-up)
#   --gpus all             -> CUDA for YOLO (RTX 5090)
#   --shm-size 2g          -> Fast DDS shared memory between nodes (Docker's default /dev/shm is 64 MB)
#   WSLg mounts            -> RViz2 / OpenCV windows show up on the Windows desktop
#   PNP_IMAGE=pnp:pose     -> the image with FoundationPose (docker/Dockerfile.pose): the default when it exists;
#                             PNP_IMAGE=pnp:jazzy forces the plain one. Takes effect on re-creation
#   -p 127.0.0.1:8088:8088 -> the workcell page (pnp_manipulation/workcell_node.py), loopback only
#   PNP_USB=1              -> (opt-in) USB access for a RealSense plugged into THIS PC: --privileged + /dev.
#                             Attach the camera to WSL first (usbipd, docs/learning/08_real_sensors.md). Takes
#                             effect when the container is (re)created:
#                               docker rm -f pnp_dev; PNP_USB=1 bash ~/pnp_repo/docker/run_dev.sh
set -euo pipefail
NAME=pnp_dev
DOCKER=(/usr/bin/docker -H unix:///var/run/docker-pnp.sock)   # the project engine, never Docker Desktop
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
mkdir -p "$HOME/pnp_ws" "$HOME/pnp_data"
USB=()
if [ "${PNP_USB:-0}" = "1" ]; then USB=(--privileged -v /dev:/dev); fi

# the image: PNP_IMAGE, else pnp:pose when it exists (FoundationPose: a re-created container must not quietly lose it)
IMAGE="${PNP_IMAGE:-}"
if [ -z "$IMAGE" ]; then
  if [ -n "$("${DOCKER[@]}" images -q pnp:pose 2>/dev/null)" ]; then IMAGE=pnp:pose; else IMAGE=pnp:jazzy; fi
fi

if [ -z "$("${DOCKER[@]}" ps -q -f name="^${NAME}$")" ]; then
  "${DOCKER[@]}" rm -f "$NAME" >/dev/null 2>&1 || true
  # 8088: the workcell page (pnp_manipulation/workcell_node.py), http://localhost:8088 on Windows
  "${DOCKER[@]}" run -d --name "$NAME" --hostname pnp-dev "${USB[@]}" \
    --init -p 127.0.0.1:7447:7447 -p 127.0.0.1:8088:8088 --gpus all --shm-size 2g \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}" \
    -v "$REPO":/repo -v "$HOME/pnp_ws":/ws -v "$HOME/pnp_data":/data \
    -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -v /usr/lib/wsl:/usr/lib/wsl \
    --device /dev/dxg \
    -e DISPLAY -e WAYLAND_DISPLAY -e XDG_RUNTIME_DIR=/mnt/wslg/runtime-dir -e PULSE_SERVER \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    "$IMAGE" bash -c 'bash /repo/setup/zenoh_container.sh & exec sleep infinity' >/dev/null
fi

if [ $# -gt 0 ]; then
  # With a terminal attached, allocate a TTY (-t): then Ctrl+C reaches the command INSIDE the container.
  # Without -t, Ctrl+C only kills the docker client and e.g. a `ros2 launch` keeps running in the
  # background (duplicate controller managers fighting over the robot!).
  if [ -t 0 ] && [ -t 1 ]; then TTY=(-it); else TTY=(-i); fi
  exec "${DOCKER[@]}" exec "${TTY[@]}" "$NAME" bash -ic "$*"
else
  exec "${DOCKER[@]}" exec -it "$NAME" bash
fi
