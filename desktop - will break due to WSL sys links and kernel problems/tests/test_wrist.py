import xml.etree.ElementTree as ET

import numpy as np

from pnp_core.catalog import load_yaml
from pnp_core.geometry import R_LINK_OPTICAL
from pnp_core.wrist import T_parent_link, T_parent_optical, urdf_snippet

CAM = load_yaml("config/camera_d405.yaml")


def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def test_optical_axis_crosses_the_tool_axis_ahead_of_the_fingers():
    T = T_parent_optical(CAM)
    p, z = T[:3, 3], T[:3, 2]
    t = -p[0] / z[0]                                     # where the viewing ray reaches x = 0 (the tool axis)
    hit = p + t * z
    assert abs(hit[1]) < 1e-9 and 0.15 < hit[2] < 0.22   # beyond the TCP (0.1034 m): the object at pre-grasp
    assert np.allclose(T[:3, 0], [0, 1, 0])              # image x = finger travel direction
    assert T[:3, 1] @ np.array([-1.0, 0, 0]) > 0.8       # image down = toward the tool axis (fingers at the bottom)


def test_link_and_optical_are_consistent():
    assert np.allclose(T_parent_link(CAM)[:3, :3] @ R_LINK_OPTICAL, T_parent_optical(CAM)[:3, :3])


def test_urdf_snippet_parses_and_reproduces_the_mount():
    root = ET.fromstring(f"<robot>{urdf_snippet(CAM)}</robot>")
    j = root.find("joint[@name='wrist_camera_link_housing_joint']")
    assert j.find("parent").get("link") == "fr3_hand"
    o = j.find("origin")
    R = rpy_to_R(*map(float, o.get("rpy").split()))
    assert np.allclose(R, T_parent_link(CAM)[:3, :3], atol=1e-5)
    assert np.allclose(list(map(float, o.get("xyz").split())), CAM["mount"]["xyz"], atol=1e-6)
    assert root.find("link[@name='wrist_camera_link_housing']/collision/geometry/box") is not None
    # the camera frames themselves come from camera_tf_node (calibration), never from the URDF
    assert root.find("link[@name='wrist_camera_link']") is None and R_LINK_OPTICAL.shape == (3, 3)


def test_so101_mount_is_the_urdf_camera_joint_plus_housing_to_optical():
    """config/robot_so101.yaml wrist_camera_mount = the D405 URDF's d405_camera_mount joint (housing frame: z = view,
    glass at z = 0) x (180 deg about z, left imager 9 mm toward image-left, 4 mm behind the glass)."""
    import pytest
    from pnp_core.catalog import load_robot, repo_root
    from pnp_core.geometry import make_transform
    from pnp_core.wrist import housing_in_description, housing_link, wrist_camera
    robot = load_robot("so101")
    urdf = repo_root() / robot["description"]["urdf"]
    if not urdf.exists():
        pytest.skip("SO-101 D405 description not present")
    j = ET.parse(urdf).getroot().find("joint[@name='d405_camera_mount']")
    o = j.find("origin")
    T_gh = make_transform(rpy_to_R(*map(float, o.get("rpy").split())), list(map(float, o.get("xyz").split())))
    T_ho = make_transform(np.diag([-1.0, -1.0, 1.0]), [0.009, 0.0, -0.004])
    cam = wrist_camera(robot)
    assert cam["mount"]["parent"] == j.find("parent").get("link") == "gripper_link"
    assert np.allclose(T_parent_optical(cam), T_gh @ T_ho, atol=1e-5)
    assert housing_in_description(cam) and housing_link(cam) == j.find("child").get("link")
    # the view goes toward the jaws (gripper_link -z) and the image's down axis toward the tool axis
    assert T_parent_optical(cam)[2, 2] < -0.8
