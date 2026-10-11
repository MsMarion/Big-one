#!/usr/bin/env bash
# Build the pnp ROS 2 image. Run inside WSL Ubuntu-24.04 (uses the distro's own Docker Engine).
set -euo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
/usr/bin/docker -H unix:///var/run/docker-pnp.sock build -t pnp:jazzy -f "$REPO/docker/Dockerfile" "$REPO" "$@"
