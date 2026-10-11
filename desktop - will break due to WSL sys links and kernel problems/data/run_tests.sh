#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash
cd /repo
python3 -m pytest -q -p no:cacheprovider tests "$@" 2>&1 | tail -25
python3 -m flake8 --select=F,E9 pnp_core sim tools tests setup ros2_ws/src/pnp_bringup ros2_ws/src/pnp_perception ros2_ws/src/pnp_manipulation && echo "flake8 (pyflakes): clean"
