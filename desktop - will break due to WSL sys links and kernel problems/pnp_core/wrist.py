"""Wrist (eye-in-hand) camera mount: one place that turns config/camera_d405.yaml into transforms, used by the sim
(the camera prim under the hand), the robot description (a fixed camera link in the URDF -> TF + collision) and
tests. docs/learning/14_wrist_camera.md

Frames (REP-103 / realsense2_camera convention):
  wrist_camera_link                  x forward (viewing direction), y left, z up
  wrist_camera_color_optical_frame   z forward, x right, y down

The camera file's `mount` is the FR3's (a tilt on fr3_hand). A robot profile overrides it with `wrist_camera_mount`
(wrist_camera() below), in either form:
  tilt   {parent, xyz, tilt_deg, body_size}   the FR3: beside the hand, tilted toward the tool axis; we add a box
                                              housing to the URDF (urdf_snippet)
  pose   {parent, xyz, rpy, housing_link}     the optical frame's pose in `parent`, e.g. from a CAD mount (the SO-101's
                                              D405 holder); the robot's URDF already has the camera's geometry as
                                              `housing_link`, so nothing is added
"""

from __future__ import annotations

import numpy as np

from .catalog import load_yaml
from .geometry import R_LINK_OPTICAL, make_transform, rpy_to_matrix


def wrist_camera(robot_cfg: dict) -> dict | None:
    """The robot's wrist camera: its camera file (robot profile `wrist_camera`) with the profile's
    `wrist_camera_mount` as the mount, or None without a wrist camera."""
    path = robot_cfg.get("wrist_camera")
    if not path:
        return None
    cam = load_yaml(path)
    if robot_cfg.get("wrist_camera_mount"):
        cam["mount"] = dict(robot_cfg["wrist_camera_mount"])
    # a scene can carry a real unit's intrinsics (the plate twin: the real D405's fx/fy/cx/cy) -> the sim renders with them
    try:
        from .catalog import load_scene
        cam.update(load_scene().get("wrist_camera_intrinsics") or {})
    except (OSError, KeyError):
        pass
    return cam


def T_parent_optical(cam_cfg: dict) -> np.ndarray:
    """Optical frame in the mount's parent link.
    pose form: as given. tilt form (fr3_hand): looking along +z (toward the fingertips), tilted by tilt_deg toward
    the tool axis (the camera sits at +x, so the view leans to -x); image x = hand y."""
    m = cam_cfg["mount"]
    if "rpy" in m:
        return make_transform(rpy_to_matrix(*m["rpy"]), m["xyz"])
    th = np.radians(m["tilt_deg"])
    z = np.array([-np.sin(th), 0.0, np.cos(th)])
    x = np.array([0.0, 1.0, 0.0])
    y = np.cross(z, x)                                      # image "down" = toward the tool axis (fingers there)
    return make_transform(np.column_stack([x, y, z]), m["xyz"])


def T_parent_link(cam_cfg: dict) -> np.ndarray:
    """camera_link in the parent link: optical = link @ R_LINK_OPTICAL."""
    T = T_parent_optical(cam_cfg).copy()
    T[:3, :3] = T[:3, :3] @ R_LINK_OPTICAL.T
    return T


def housing_in_description(cam_cfg: dict) -> bool:
    """Does the robot's own URDF model the camera (pose form)? Then there is no housing box to add."""
    return "housing_link" in cam_cfg["mount"]


def housing_link(cam_cfg: dict) -> str:
    return cam_cfg["mount"].get("housing_link") or cam_cfg["frames"]["link"] + "_housing"


def urdf_snippet(cam_cfg: dict) -> str:
    """URDF link + fixed joint for the wrist camera HOUSING (collision + visual box, at the nominal mount): MoveIt
    keeps it out of obstacles, the obstacle filter removes its points. The camera FRAMES are not in the URDF: they
    come from the calibration (pnp_perception/camera_tf_node.py), so a new calibration applies without a restart.
    Tilt form only (the pose form's housing is already in the robot's URDF)."""
    m = cam_cfg["mount"]

    def rpy(R):                                              # URDF wants roll-pitch-yaw (fixed axes x, y, z)
        return (np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-np.clip(R[2, 0], -1, 1)), np.arctan2(R[1, 0], R[0, 0]))

    def origin(T):
        return (f'<origin xyz="{T[0, 3]:.6f} {T[1, 3]:.6f} {T[2, 3]:.6f}" '
                f'rpy="{" ".join(f"{a:.6f}" for a in rpy(T[:3, :3]))}"/>')
    T_pl = T_parent_link(cam_cfg)
    sx, sy, sz = m["body_size"]
    box = f'<geometry><box size="{sx} {sy} {sz}"/></geometry>'
    # the housing sits BEHIND the lens: shift it back by half its depth along camera_link x
    body = f'<origin xyz="{-sx / 2:.4f} 0 0" rpy="0 0 0"/>'
    h = housing_link(cam_cfg)
    return (f'<link name="{h}"><visual>{body}{box}<material name="d405_grey"><color rgba="0.3 0.3 0.32 1"/>'
            f'</material></visual><collision>{body}{box}</collision></link>'
            f'<joint name="{h}_joint" type="fixed"><parent link="{m["parent"]}"/><child link="{h}"/>'
            f'{origin(T_pl)}</joint>')
