#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
for t in /isaac_joint_states /joint_states; do printf "%-22s " $t; timeout 5 ros2 topic hz $t --window 10 2>/dev/null | grep -m1 -o "average rate: [0-9.]*" || echo "NO DATA"; done
