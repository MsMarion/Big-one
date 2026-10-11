# ROS 2 side of the project: perception, planning, control, tools.
# The SAME image is used for development (docker/run_dev.sh, inside WSL) and on the robot PC.
# Source code is NOT baked in during development: the repo is bind-mounted at /repo.
#
# Build (from WSL):  bash ~/pnp_repo/docker/build.sh
FROM osrf/ros:jazzy-desktop

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# --- ROS packages (apt) ------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-dev-tools python3-venv python3-pip git rsync curl \
      ros-jazzy-moveit ros-jazzy-moveit-py ros-jazzy-moveit-planners-chomp \
      ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager \
      ros-jazzy-joint-trajectory-controller ros-jazzy-gripper-controllers \
      ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-joint-state-publisher \
      ros-jazzy-tf2-ros ros-jazzy-tf2-tools ros-jazzy-tf-transformations \
      ros-jazzy-cv-bridge ros-jazzy-image-geometry ros-jazzy-vision-msgs ros-jazzy-message-filters \
      ros-jazzy-rosbag2-storage-mcap ros-jazzy-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# Phase 5B / 6: depth -> point cloud (depth_image_proc) for the Octomap; JPEG / lossless-PNG image transport
# (image_transport_plugins) to receive a REAL camera over the network (Jetson -> PC) and decompress it here.
# moveit_ros_perception = MoveIt's Octomap updaters (PointCloudOctomapUpdater); NOT part of ros-jazzy-moveit.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-depth-image-proc ros-jazzy-image-transport-plugins ros-jazzy-moveit-ros-perception usbutils \
    && rm -rf /var/lib/apt/lists/*

# Optional packages: not every one is released for Jazzy on apt. Missing ones are listed in
# /opt/pnp_missing_apt.txt and built from source in the workspace (ros2_ws/pnp.repos).
RUN apt-get update && : > /opt/pnp_missing_apt.txt && \
    for pkg in ros-jazzy-topic-based-ros2-control ros-jazzy-franka-description \
               ros-jazzy-realsense2-camera ros-jazzy-realsense2-description; do \
      apt-get install -y --no-install-recommends "$pkg" || echo "$pkg" >> /opt/pnp_missing_apt.txt; \
    done && rm -rf /var/lib/apt/lists/* && echo "missing apt pkgs:" && cat /opt/pnp_missing_apt.txt

# --- zenoh bridge: carries ROS 2 topics between this container and Isaac Sim on Windows over TCP ---
# Must match the Windows binary version (setup/zenoh_windows.ps1). Checksum = GitHub release digest.
ARG ZENOH_VER=1.10.1
ARG ZENOH_SHA256=f3ea8ac4345be22716925269bfd3e350e7cb36a8f056e835468c8299341d29b0
RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/* && \
    curl -fsSL -o /tmp/zenoh.zip \
      "https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases/download/${ZENOH_VER}/zenoh-plugin-ros2dds-${ZENOH_VER}-x86_64-unknown-linux-gnu-standalone.zip" && \
    echo "${ZENOH_SHA256}  /tmp/zenoh.zip" | sha256sum -c - && \
    mkdir -p /opt/zenoh && unzip -q /tmp/zenoh.zip -d /opt/zenoh && rm /tmp/zenoh.zip && \
    ln -s /opt/zenoh/zenoh-bridge-ros2dds /usr/local/bin/zenoh-bridge-ros2dds

# --- Python venv for perception/training (sees the ROS system packages) -------------------------
# RTX 5090 (Blackwell, sm_120) needs CUDA 12.8+ wheels. numpy<2 keeps cv_bridge working.
RUN python3 -m venv --system-site-packages /opt/pnp_venv && \
    /opt/pnp_venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/pnp_venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
COPY requirements-perception.txt /tmp/requirements-perception.txt
RUN /opt/pnp_venv/bin/pip install --no-cache-dir -r /tmp/requirements-perception.txt

# Cyclone DDS (its own layer: a rebuild keeps everything above): ROS services across the zenoh link to the Jetson
# only work with the bridge's own DDS (config/cyclonedds.xml, setup/ros_env.sh)
RUN apt-get update && apt-get install -y --no-install-recommends ros-jazzy-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

# --- Environment -------------------------------------------------------------------------------
ENV PATH=/opt/pnp_venv/bin:$PATH \
    PYTHONPATH=/repo \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=42 \
    PNP_REPO=/repo PNP_WS=/ws PNP_DATA=/data
RUN echo 'source /repo/setup/ros_env.sh' >> /root/.bashrc
WORKDIR /ws
CMD ["bash"]
