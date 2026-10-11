#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
cd /repo
echo "=== fixed"
python3 tools/calibrate.py fixed --poses 12 > /tmp/fixed_out.txt 2>&1; tail -12 /tmp/fixed_out.txt; grep -c "not reachable" /tmp/fixed_out.txt
python3 tools/set_reality.py --truth
