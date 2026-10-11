#!/usr/bin/env bash
# Sanity-check the pnp:jazzy image: ROS packages, CUDA, numpy/cv_bridge compatibility.
#   bash ~/pnp_repo/docker/check_image.sh
set -uo pipefail
/usr/bin/docker -H unix:///var/run/docker-pnp.sock run --rm --gpus all pnp:jazzy bash -c '
source /opt/ros/jazzy/setup.bash
echo "== optional apt packages missing (build from source):"; cat /opt/pnp_missing_apt.txt
echo "== python stack"
python3 - <<EOF
import numpy, torch, cv2, ultralytics
from cv_bridge import CvBridge
print("numpy", numpy.__version__, "| torch", torch.__version__, "| cv2", cv2.__version__, "| ultralytics", ultralytics.__version__)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
x = torch.ones(1000, 1000, device="cuda")
print("cuda matmul ok:", float((x @ x).sum()) == 1e9)
CvBridge(); print("cv_bridge ok")
EOF
echo "== key ROS packages present"
ros2 pkg list | grep -E "^(moveit_py|moveit_ros_move_group|gripper_controllers|joint_trajectory_controller|topic_based_ros2_control|franka_description|realsense2_camera|vision_msgs|rosbag2_storage_mcap)$"
'
