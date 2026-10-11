#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
ros2 node list 2>/dev/null | grep -i workcell || echo "no workcell node"
grep -i -E "workcell" /data/logs_bringup.log | tail -15 | cut -c1-250
curl -s -m 5 http://127.0.0.1:8088/api/world | head -c 600; echo
