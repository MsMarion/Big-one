# SO101 UVC mount with D435i camera

This package exports the current Fusion assembly in metres. Open `so101_uvc_d435i_mount.urdf` with the adjacent `meshes` directory.

The `d435i_tilt` joint rotates about the camera's horizontal width axis through the M4 pivot. URDF joint position 0 matches the Fusion pose at export (21.709 degrees from Fusion's original horizontal camera pose). The camera, tilt plate, and two M3 screws move together; the UVC mount, fork, M4 bolt, washers, and locknut stay fixed.

The joint is declared `continuous` because no physical rotation stops were measured. Collision meshes cover the five main components; small fasteners are visual only. Mass, inertia, calibrated optical frame, camera intrinsics, and hardware drive parameters are not included.
