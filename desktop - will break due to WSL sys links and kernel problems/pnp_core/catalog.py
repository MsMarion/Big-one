"""Typed-ish access to the YAML configs in config/. Keeps file names and keys in one place."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from .perception_math import Intrinsics


def repo_root() -> Path:
    env = os.environ.get("PNP_REPO")
    return Path(env) if env else Path(__file__).resolve().parents[1]


def load_yaml(rel_or_abs) -> dict:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = repo_root() / p
    return yaml.safe_load(p.read_text())


def load_object(name: str) -> dict:
    objects = load_yaml("config/objects.yaml")
    if name not in objects:
        raise KeyError(f"unknown object {name!r}; known: {sorted(objects)}")
    return objects[name]


# --- the ACTIVE cell ------------------------------------------------------------------------------------------
# A scene file describes a whole cell: which robot (`robot:`, default fr3), where the fixed camera hangs
# (`camera:` overrides of config/camera_d435i.yaml), the table, the objects. Everything follows the active scene:
#   1. env PNP_SCENE (bringup.launch.py sets it for every node; run_sim.py for the sim)
#   2. the file /data/pnp_active_scene (written by bringup: tools started later in the container find it)
#   3. config/scenes/basic.yaml (the FR3 table)
ACTIVE_SCENE_FILE = Path("/data/pnp_active_scene")
DEFAULT_SCENE = "config/scenes/basic.yaml"


def active_scene_path() -> str:
    env = os.environ.get("PNP_SCENE")
    if env:
        return env
    try:
        if ACTIVE_SCENE_FILE.exists():
            return ACTIVE_SCENE_FILE.read_text().strip() or DEFAULT_SCENE
    except OSError:
        pass
    return DEFAULT_SCENE


def load_scene(path: str | None = None) -> dict:
    return load_yaml(path or active_scene_path())


def robot_name(scene: dict | None = None) -> str:
    """The robot of a scene (default: the active scene). Env PNP_ROBOT overrides (tests, one-off tools)."""
    return os.environ.get("PNP_ROBOT") or (scene if scene is not None else load_scene()).get("robot", "fr3")


def load_robot(name: str | None = None) -> dict:
    """The robot profile config/robot_<name>.yaml (default: the active scene's robot)."""
    return load_yaml(f"config/robot_{name or robot_name()}.yaml")


def cell_name(scene: dict | None = None) -> str:
    """Which physical cell (robot + camera layout) a scene belongs to: calibrations are per cell."""
    return (scene if scene is not None else load_scene()).get("cell", "fr3_table")


def load_fixed_camera(scene: dict | None = None) -> dict:
    """config/camera_d435i.yaml with the scene's `camera:` overrides (e.g. a different mount) merged in."""
    cam = load_yaml("config/camera_d435i.yaml")
    over = (scene if scene is not None else load_scene()).get("camera") or {}
    for k, v in over.items():
        cam[k] = {**cam[k], **v} if isinstance(v, dict) and isinstance(cam.get(k), dict) else v
    m = cam.get("mount") or {}
    if m.get("from_calibration"):
        # the mount IS a calibration result (the sim twin of a real cell: config/scenes/so101_plate.yaml): its full
        # pose; look_at = where the view axis meets the desk (for tools that only use a look-at, e.g. the SDG jitter)
        c = load_yaml(m["from_calibration"])["T_parent_optical"]
        p = np.array(c["xyz"], float)
        q = np.array(c["quat_xyzw"], float)
        x, y, z, w = q
        zax = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])     # the optical z axis
        t = (scene if scene is not None else load_scene()).get("table") or {}
        top = t["center"][2] + t["size"][2] / 2 if t else 0.0
        hit = p + zax * (top - p[2]) / zax[2] if zax[2] < -1e-6 else p + zax * 0.5
        cam["mount"] = {**m, "position": p.tolist(), "quat_xyzw": q.tolist(), "look_at": hit.round(4).tolist()}
    return cam


def calibration_targets_path(scene: dict | None = None) -> str:
    """The calibration targets (boards, AprilTags) of the active cell (scene key calibration_targets)."""
    return (scene if scene is not None else load_scene()).get("calibration_targets", "config/calibration_targets.yaml")


def calibration_dir(scene: dict | None = None) -> Path:
    """Calibration files of the active cell: config/calibration/ for the FR3 table (as before), else a subfolder."""
    cell = cell_name(scene)
    return repo_root() / "config" / "calibration" / ("" if cell == "fr3_table" else cell)


def load_camera(name: str = "d435i") -> dict:
    return load_fixed_camera() if name == "d435i" else load_yaml(f"config/camera_{name}.yaml")


def camera_intrinsics(cam: dict) -> Intrinsics:
    return Intrinsics(cam["fx"], cam["fy"], cam["cx"], cam["cy"])


def camera_matrix(cam: dict) -> np.ndarray:
    return np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1.0]])
