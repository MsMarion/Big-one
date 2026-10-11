#!/usr/bin/env bash
# Build pnp:pose (pnp:jazzy + FoundationPose environment). Run inside WSL Ubuntu-24.04.
# Takes a while: CUDA toolkit (~4 GB) and PyTorch3D compiles from source (~20-30 min).
# To use it, recreate the dev container from it:  PNP_IMAGE=pnp:pose bash ~/pnp_repo/docker/run_dev.sh
set -euo pipefail
REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
/usr/bin/docker -H unix:///var/run/docker-pnp.sock build -t pnp:pose -f "$REPO/docker/Dockerfile.pose" "$REPO" "$@"
