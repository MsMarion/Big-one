#!/usr/bin/env bash
# Entry of the Jetson ROBOT container (jetson/docker-compose.robot.yml): build the workspace from the mounted repo
# (only when the sources changed), then start the robot role of the stack.
#   PNP_SCENE      the scene FILE (default config/scenes/so101_workcell.yaml; the same variable every tool reads)
#   PNP_HW_MOCK    true: no servo board needed (a simulated bus; the first test of everything else)
#   PNP_BUILD_ONLY 1: only build the workspace, start nothing
#   PNP_CAMERA     wrist (the D405 only, default) | local (D405 + a D435i on this Jetson) | none
set -e
source /repo/setup/ros_env.sh
mkdir -p /repo/ros2_ws/src/external /ws
# --symlink-install: Python edits are live; rebuild only when a package's manifest / build files change (a new node
# is a new setup.py entry). The repo is the Jetson's git clone (~/pnp, jetson/sync_to_jetson.sh).
stamp=$(cd /repo/ros2_ws && find src -name package.xml -o -name setup.py -o -name CMakeLists.txt -o -name '*.launch.py'         | sort | xargs cat | md5sum | cut -d' ' -f1)
if [ ! -f /ws/install/setup.bash ] || [ "$stamp" != "$(cat /ws/.built 2>/dev/null)" ]; then
  echo "[robot] building the workspace (first time: a few minutes)"
  vcs import --skip-existing /repo/ros2_ws/src/external < /repo/ros2_ws/pnp.repos
  # the robot role's nodes live in all three; 2 workers: 8 GB of RAM shared with the desktop
  pnp_build --packages-up-to pnp_bringup pnp_manipulation pnp_perception --parallel-workers 2
  echo "$stamp" > /ws/.built
fi
source /ws/install/setup.bash
# PNP_BUILD_ONLY=1: build and stop (no devices opened): docker compose ... run --rm -e PNP_BUILD_ONLY=1 robot
[ "${PNP_BUILD_ONLY:-0}" = 1 ] && { echo "[robot] workspace built"; exit 0; }
exec ros2 launch pnp_bringup bringup.launch.py role:=robot sim:=false hardware:=real manipulation:=true \
  camera:="${PNP_CAMERA:-wrist}" hw_mock:="${PNP_HW_MOCK:-false}" \
  scene:="${PNP_SCENE:-config/scenes/so101_workcell.yaml}" use_rviz:=false
