#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
timeout 6 ros2 topic echo --once /perception/detections 2>/dev/null | grep -E "class_id|score|^    x:|^    y:" | head -20
grep -E "detector ready|focus" /data/logs_bringup.log | tail -4 | cut -c1-220
