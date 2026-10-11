#!/usr/bin/env bash
# Inside the pnp container: fetch pinned third-party sources (ros2_ws/pnp.repos) and build the workspace.
#   bash ~/pnp_repo/docker/run_dev.sh "bash /repo/setup/ws_bootstrap.sh"
set -e
source /repo/setup/ros_env.sh
mkdir -p /repo/ros2_ws/src/external
vcs import --skip-existing /repo/ros2_ws/src/external < /repo/ros2_ws/pnp.repos
# Only what we need: building all of franka_ros2 would require libfranka (real-robot drivers, Phase 6).
pnp_build --packages-up-to pnp_bringup
echo "workspace ready: ros2 launch pnp_bringup bringup.launch.py"
