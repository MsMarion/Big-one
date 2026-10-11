#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
cd /repo
python3 tools/set_reality.py misplaced_camera
echo "=== locate"
python3 tools/calibrate.py locate
echo "=== fixed"
python3 tools/calibrate.py fixed --poses 12 2>&1 | tail -14
python3 tools/set_reality.py --truth
