#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
cd /repo
f=$(ls -t data/calibration/arm_accuracy_*.json | head -1)
python3 tools/fit_joint_offsets.py $f "$@" | tail -4
cat config/calibration/so101_desk/joint_offsets_sim_typical.yaml 2>/dev/null
