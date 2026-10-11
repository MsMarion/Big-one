"""Every YAML config and RViz layout must parse. RViz does not report a broken file: it opens an empty window
(an unquoted ': ' inside a display name did exactly that to sensors.rviz)."""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
FILES = sorted([*REPO.glob("config/**/*.yaml"), *REPO.glob("ros2_ws/src/*/config/**/*.yaml"),
                *REPO.glob("ros2_ws/src/*/rviz/*.rviz")])


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_parses(path):
    assert yaml.safe_load(path.read_text()) is not None


def test_rviz_layouts_have_displays():
    for p in REPO.glob("ros2_ws/src/*/rviz/*.rviz"):
        assert yaml.safe_load(p.read_text())["Visualization Manager"]["Displays"], p.name


@pytest.mark.parametrize("layout", ["moveit.rviz", "sensors.rviz"])
@pytest.mark.parametrize("robot", ["fr3", "so101"])
def test_rviz_layout_follows_the_robot(layout, robot):
    """RViz draws everything in the fixed frame: the FR3's fr3_link0 on an SO-101 = every display a TF error."""
    from pnp_core.catalog import load_robot
    from pnp_core.robot_description import rviz_config
    cfg = load_robot(robot)
    text = rviz_config((REPO / "ros2_ws/src/pnp_bringup/rviz" / layout).read_text(), cfg)
    out = yaml.safe_load(text)
    assert out["Visualization Manager"]["Global Options"]["Fixed Frame"] == cfg["frames"]["base"]
    assert out["Visualization Manager"]["Displays"]
    if robot != "fr3":
        assert "fr3" not in text
    if not cfg.get("wrist_camera"):
        assert "/wrist_camera/" not in text
    elif layout == "sensors.rviz":
        assert "/wrist_camera/" in text
