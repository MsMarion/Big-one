# SO-101 with D405 wrist camera

`so101_d405_wrist.urdf` is a self-contained kinematic and collision model. Keep
the URDF and `assets/` directory together so relative mesh paths resolve.
It does not require ROS 2 or a Fusion export add-in.

The six revolute joints are `shoulder_pan`, `shoulder_lift`, `elbow_flex`,
`wrist_flex`, `wrist_roll`, and `gripper`. They retain the axes and joint limits
from [The Robot Studio's SO-101 URDF](https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf).
The D405 holder replaces the stock wrist-roll follower geometry on
`gripper_link`. The D405 camera is `d405_camera_link`, connected by the fixed
`d405_camera_mount` joint, so it follows wrist roll. The original fixed
`gripper_frame_joint` remains. The source's control transmissions were removed;
only robot geometry and kinematics are included.

The D405 camera's collision geometry is a bounding box. Its visual geometry
uses official camera CAD converted to STL. The holder uses The Robot Studio's
D405 wrist mount STL. All STL coordinates are in metres in this package.
The camera pose was measured in the Fusion replacement assembly and composed
with the source URDF wrist mesh pose.

**Camera import note:** read [CAMERA_IMPORT_NOTES.md](CAMERA_IMPORT_NOTES.md)
before assigning camera properties. This URDF contains a nominal CAD housing
pose, not the D405 unit's optical calibration or verified optical frame.

This URDF is suitable for inspecting link motion and approximate space
occupancy. The gripper inertial properties still come from the stock SO-101
URDF, so dynamics are not calibrated for the new holder and camera. The D435i
overhead camera is separate from the moving robot and is not in this URDF.

The saved Fusion design **SO101 D405 Wrist Replacement Assembly** is still a
static detailed CAD assembly. Its imported STEP hierarchy has not been
reorganized into moving links or given Fusion joints. This URDF's joint tree
comes from the official SO-101 model, with the replacement geometry added.

Rebuild with `python tools/build_so101_d405_urdf.py` from the workspace root.
See `UPSTREAM_LICENSE` for The Robot Studio assets.
