# D405 camera data to check at import

The URDF currently defines the **D405 housing mesh and its nominal mount pose**.
It does **not** contain verified sensor calibration. In particular,
`d405_camera_link` is the CAD model frame, not a measured depth or color optical
frame. Do not use its origin as the optical center without checking it against
the physical camera.

When connecting the camera model to a simulator or vision application, record
these values from the **specific D405 unit and active stream profiles**:

| Item | Record and compare |
| --- | --- |
| Device identity | Model, serial number, firmware version, and SDK version. |
| Stream profile | Depth and color width, height, format, and frame rate actually selected. |
| Intrinsics | `fx`, `fy`, `ppx`, `ppy`, distortion model, and coefficients for each selected stream. Do not reuse one resolution's intrinsics for another. |
| Sensor extrinsics | Depth-to-color transform reported by the device, in metres. Keep its direction explicit. |
| Depth units | Actual depth scale in metres per raw unit, plus any processing that changes depth or alignment. |
| Mount extrinsic | Rigid transform from the wrist/CAD housing frame to the **depth optical frame**. Measure or calibrate it on the assembled robot, including the camera's location within its housing. |

The RealSense SDK's `rs-enumerate-devices -c` reports calibration and stream
information for a connected unit. Its depth sensor API reports the active depth
scale. For each camera stream, use the calibration associated with the actual
profile in use. If the camera, resolution, mount, or calibration changes,
recheck the corresponding values before reusing a saved configuration.

For a realistic model, use the device's calibrated intrinsics and sensor
extrinsics, then verify the wrist-to-optical transform using an observed target
at several wrist angles and working distances. The nominal CAD mount transform
in this URDF is a starting estimate; it does not capture print tolerances,
assembly offsets, or camera optical center placement. Keep these measured
sensor parameters in a separate calibration file tied to the serial number and
profile. URDF itself is for link geometry and joint placement, and has no
standard fields for all RealSense calibration data.

Official references:

- [RealSense device enumeration and `-c` flag](https://github.com/realsenseai/librealsense/blob/master/tools/enumerate-devices/readme.md)
- [RealSense SDK projection, intrinsics, and stream extrinsics](https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0)
- [RealSense SDK depth scale and intrinsic API examples](https://github.com/realsenseai/librealsense/blob/master/examples/sensor-control/api_how_to.h)
- [RealSense D400 series datasheet](https://dev.realsenseai.com/download/42003/)
