#!/usr/bin/env bash
# One-time prep of the WSL2 Ubuntu-24.04 distro as a *Docker host* for the ROS 2 side.
# ROS itself lives in our image (docker/Dockerfile), not in this distro.
#
#   wsl -d Ubuntu-24.04 -u root -- bash "/mnt/c/Users/<you>/Desktop/Robotics Pick and Place/issac standard/setup/wsl_host_setup.sh"
#
# Why Docker Engine *inside* this distro instead of Docker Desktop:
#   With WSL mirrored networking, `--network host` here means "Windows' network", so ROS 2/DDS in
#   the container talks to Isaac Sim on Windows like two local processes. Docker Desktop's containers
#   sit behind NAT in its own VM, which breaks DDS (tested: see docs/learning/00_setup.md).
# Idempotent: safe to re-run.
set -euo pipefail

# pwd -P: the PHYSICAL /mnt/c/... path. If this script is run via the ~/pnp_repo symlink, a logical
# path would make the `ln -sfn` below point the symlink at itself.
REPO_WIN="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "== 1. Docker Engine (docker-ce) from Docker's apt repo"
$SUDO apt-get update
$SUDO apt-get install -y ca-certificates curl gnupg
$SUDO install -m 0755 -d /etc/apt/keyrings
if [ ! -f /etc/apt/keyrings/docker.asc ]; then
  $SUDO curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  $SUDO chmod a+r /etc/apt/keyrings/docker.asc
fi
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null

echo "== 2. NVIDIA Container Toolkit (GPU inside containers; uses the Windows driver via WSL)"
if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | $SUDO gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
fi
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

$SUDO apt-get update
$SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
  nvidia-container-toolkit
$SUDO nvidia-ctk runtime configure --runtime=docker

# Own socket: Docker Desktop's "WSL integration" takes over /var/run/docker.sock in the default distro,
# which silently points `docker` at Docker Desktop's engine instead of this one. Our engine listens on
# /var/run/docker-pnp.sock instead; project scripts use docker/pnp_docker.sh to talk to it.
$SUDO mkdir -p /etc/systemd/system/docker.service.d
$SUDO tee /etc/systemd/system/docker.service.d/pnp-socket.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H unix:///var/run/docker-pnp.sock --containerd=/run/containerd/containerd.sock
EOF
$SUDO systemctl disable --now docker.socket 2>/dev/null || true
$SUDO systemctl daemon-reload
$SUDO systemctl enable docker
$SUDO systemctl restart docker

echo "== 3. Kernel socket buffer limits for DDS (camera images are ~1 MB = many UDP fragments)"
$SUDO tee /etc/sysctl.d/60-pnp-dds.conf >/dev/null <<'EOF'
net.core.rmem_max = 67108864
net.core.wmem_max = 67108864
net.ipv4.ipfrag_high_thresh = 134217728
net.ipv4.ipfrag_time = 3
EOF
$SUDO sysctl --system >/dev/null

echo "== 4. Workspace links and data dirs"
# The Windows repo path contains spaces, which some ROS build tools dislike: use a space-free symlink.
ln -sfn "$REPO_WIN" "$HOME/pnp_repo"
mkdir -p "$HOME/pnp_data/bags" "$HOME/pnp_data/runs" "$HOME/pnp_ws"

echo "== 5. Check"
/usr/bin/docker -H unix:///var/run/docker-pnp.sock --version
/usr/bin/docker -H unix:///var/run/docker-pnp.sock info --format 'runtimes: {{range $k, $v := .Runtimes}}{{$k}} {{end}}'
echo "== Done. Next: bash ~/pnp_repo/docker/build.sh"
