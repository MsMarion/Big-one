#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
cd /repo
f=$(ls -t data/calibration/arm_accuracy_*.json | head -1)
for r in truth fixed_camera configured_spot free; do echo "=== $r"; python3 tools/fit_joint_offsets.py $f --reference $r "$@" | tail -9; done
