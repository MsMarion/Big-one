#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
for m in aiohttp fastapi websockets flask tornado; do python3 -c "import $m; print(\"$m\", $m.__version__)" 2>/dev/null || echo "$m: no"; done
python3 -c "from interactive_markers import InteractiveMarkerServer, MenuHandler; print(\"interactive_markers ok\")" 2>&1 | tail -1
python3 -c "import cv2; print(\"cv2\", cv2.__version__)"
