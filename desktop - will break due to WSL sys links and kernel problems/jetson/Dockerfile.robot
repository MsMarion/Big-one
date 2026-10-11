# The ROBOT side on a Jetson (Orin Nano Super 8 GB, JetPack 6.2 = L4T r36.4, arm64): the SO-101's servo bridge,
# ros2_control, MoveIt, the wrist D405 + its TF, the wrist flat map. docs/learning/22_jetson_d405.md
#
# ROS 2 JAZZY like the PC (the same code and generated configs on both; Humble would need its own testing). Jazzy is
# Ubuntu 24.04; the Jetson host is 22.04 -- fine in a container (it shares only the kernel), and nothing here needs
# the GPU: drivers, control, planning and the map fusion are CPU work. (A detector on the Jetson later would use an
# L4T CUDA base instead.)
#
# Build ON the Jetson (arm64; ~20-40 min the first time, mostly apt):
#   cd ~/pnp && docker compose -f jetson/docker-compose.robot.yml build
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-dev-tools git python3-pip python3-serial python3-scipy python3-opencv python3-yaml usbutils \
      ros-jazzy-moveit-ros-move-group ros-jazzy-moveit-planners-ompl ros-jazzy-pilz-industrial-motion-planner \
      ros-jazzy-moveit-kinematics ros-jazzy-moveit-simple-controller-manager ros-jazzy-moveit-configs-utils \
      ros-jazzy-moveit-ros-planning-interface \
      ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-controller-manager \
      ros-jazzy-joint-trajectory-controller ros-jazzy-gripper-controllers \
      ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-tf2-ros ros-jazzy-tf-transformations \
      ros-jazzy-cv-bridge ros-jazzy-vision-msgs ros-jazzy-image-transport-plugins ros-jazzy-rmw-fastrtps-cpp \
    && rm -rf /var/lib/apt/lists/*

# The D405's driver. ROS's librealsense2 package uses the RSUSB (libusb) backend: no JetPack kernel patches needed.
# If apt has no arm64 build of a package, it's noted and the build continues (built from source then, see README).
RUN apt-get update && : > /opt/pnp_missing_apt.txt && \
    for pkg in ros-jazzy-librealsense2 ros-jazzy-realsense2-camera ros-jazzy-topic-based-ros2-control; do \
      apt-get install -y --no-install-recommends "$pkg" || echo "$pkg" >> /opt/pnp_missing_apt.txt; \
    done && rm -rf /var/lib/apt/lists/* && echo "missing apt pkgs:" && cat /opt/pnp_missing_apt.txt

# OpenCV >= 4.7 for the calibration (cv2.aruco.CharucoDetector; Ubuntu 24.04's python3-opencv is 4.6). A pip wheel
# (aarch64 builds exist) shadows apt's cv2 for Python only; --no-deps keeps apt's numpy 1.26 (cv_bridge, scipy).
RUN pip3 install --break-system-packages --no-deps "opencv-python-headless==4.10.0.84" &&     python3 -c "import cv2; assert hasattr(cv2.aruco, 'CharucoDetector'), cv2.__version__; print('cv2', cv2.__version__)"

# Cyclone DDS (its own layer: a rebuild keeps everything above): ROS services across the zenoh link to the PC only
# work with the bridge's own DDS (config/cyclonedds.xml)
RUN apt-get update && apt-get install -y --no-install-recommends ros-jazzy-rmw-cyclonedds-cpp && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/repo \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    ROS_DOMAIN_ID=42 \
    PNP_REPO=/repo PNP_WS=/ws PNP_DATA=/data \
    PNP_VENV_PYTHON=/usr/bin/python3
RUN echo 'source /repo/setup/ros_env.sh' >> /root/.bashrc
WORKDIR /ws
COPY jetson/robot_entry.sh /robot_entry.sh
CMD ["bash", "/robot_entry.sh"]
