#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
ros2 node list 2>&1 | head -30
echo ---
ps aux | grep -E "ros2_control_node|move_group|robot_state_pub|zenoh" | grep -v grep | awk "{print \$2, \$11, \$12, \$13}" | head
echo ---
timeout 5 ros2 topic hz /camera/camera/color/image_raw --window 5 2>&1 | grep -m1 average || echo "camera: NO DATA"
