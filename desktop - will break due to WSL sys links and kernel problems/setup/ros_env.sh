# Sourced by every shell in the pnp container (see docker/Dockerfile). Must match setup/windows_env.ps1
# (same ROS_DOMAIN_ID). Plain Fast DDS defaults: inside the container's own network namespace normal
# multicast discovery works; the zenoh bridge (setup/zenoh_container.sh) carries topics to Windows.
source /opt/ros/jazzy/setup.bash
# Fast DDS (the sim stack's tested default). PNP_RMW=rmw_cyclonedds_cpp switches: the Jetson robot role sets it
# (jetson/docker-compose.robot.yml) because ROS services across the PC <-> Jetson zenoh link need Cyclone on both ends
# (config/cyclonedds.xml). Sim work on the PC stays on Fast DDS.
export RMW_IMPLEMENTATION="${PNP_RMW:-rmw_fastrtps_cpp}"
export CYCLONEDDS_URI="file://${PNP_REPO:-/repo}/config/cyclonedds.xml"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
# Big UDP buffers so ~1 MB camera images survive (see the XML for details)
export FASTDDS_DEFAULT_PROFILES_FILE="${PNP_REPO:-/repo}/config/fastdds_container.xml"
export FASTRTPS_DEFAULT_PROFILES_FILE="$FASTDDS_DEFAULT_PROFILES_FILE"

[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash

# Build the ROS workspace: sources live in the repo (/repo/ros2_ws/src), build/install/log in /ws.
# BUILD_TESTING=OFF: third-party packages' test-only deps (e.g. ros_testing) aren't installed in the image.
pnp_build() {
  (cd /ws && colcon build --symlink-install --base-paths /repo/ros2_ws/src \
     --cmake-args -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release "$@") \
    && source /ws/install/setup.bash
}
