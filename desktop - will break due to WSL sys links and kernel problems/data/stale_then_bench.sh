#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
python3 /repo/tools/grasp_physics_check.py | tail -1
bash /repo/data/bench_look.sh "$@"
