import pytest
from fastapi.testclient import TestClient

from objsearch import db, grasp

G = {"max_open_mm": 55, "clearance_mm": 8, "min_width_mm": 10, "min_height_mm": 10, "tolerance_mm": 3}


@pytest.mark.parametrize("dims,pose,verdict", [
    ([31.8, 15.8, 11.4], None, "good"),          # LEGO 2x4 brick
    ([24.8, 14.3, 11.0], None, "good"),          # ESP-01 with header
    ([15.8, 7.8, 11.4], None, "borderline"),     # 1x2 brick: 7.8 mm across the jaws
    ([45.0, 18.0, 8.0], None, "borderline"),     # Nano: 8 mm tall
    ([68.6, 53.4, 15.0], None, "borderline"),    # Uno: tight in the jaws
    ([25.5, 18.0, 3.1], None, "no"),             # bare WROOM module: too flat
    ([15.8, 7.8, 4.9], None, "no"),              # 1x2 plate
    ([85.6, 56.5, 17.0], None, "no"),            # Pi 4: wider than the jaws open
    ([15.8, 6.2, 6.2], None, "no"),              # technic pin
    ([10.0, 10.0, 10.0], None, "good"),          # exactly at the limits
    ([40.0, 25.0, 30.0], "flat", "good"),        # lying on its largest face: 25 tall, 30 across
    ([72.9, 31.3, 66.6], None, "good"),          # webcam on its clip
])
def test_assess(dims, pose, verdict):
    a = grasp.assess(dims, pose, G)
    assert a["verdict"] == verdict, a
    assert bool(a["reasons"]) == (verdict != "good")


def test_flat_pose_and_unknown():
    a = grasp.assess([40, 25, 30], "flat", G)
    assert (a["height_mm"], a["grasp_width_mm"]) == (25, 30)
    assert grasp.assess(None)["verdict"] is None and grasp.assess([1, 2])["verdict"] is None


def test_limits_come_from_config():
    g = grasp.limits()
    assert set(G) <= set(g) and g["max_open_mm"] > g["clearance_mm"]


def test_api_grip_filter_and_pose(seeded):
    from dashboard.app import app
    client = TestClient(app)
    d = client.get("/api/objects").json()
    counts = d["summary"]["grip"]
    assert counts["good"] and counts["no"] and sum(counts.values()) == len(d["objects"])
    ok = client.get("/api/objects", params={"grip": "ok"}).json()["objects"]
    assert ok and all(o["grasp"]["verdict"] in ("good", "borderline") for o in ok)
    assert "esp32-wroom-32" not in {o["id"] for o in ok} and "lego-3001" in {o["id"] for o in ok}
    assert client.get("/api/objects", params={"grip": "maybe"}).status_code == 400
    assert client.get("/api/objects/microsd-card").json()["grasp"]["verdict"] == "no"
    # pose override: 'flat' recomputes height from the smallest dimension; only null|flat are accepted
    assert client.patch("/api/objects/sg90-servo", json={"grasp_pose": "sideways"}).status_code == 400
    o = client.patch("/api/objects/sg90-servo", json={"grasp_pose": "flat"}).json()
    assert o["grasp_pose"] == "flat" and o["grasp"]["pose"] == "flat" and o["grasp"]["height_mm"] == 12.2
    assert client.patch("/api/objects/sg90-servo", json={"grasp_pose": None}).json()["grasp"]["height_mm"] == 31


def test_gallery_grip_filter(seeded):
    from objsearch import gallery
    html = gallery.render(seeded, grip="no")
    assert "gripper: can&#x27;t pick" in html or "gripper: can't pick" in html
    assert "LEGO Brick 2 x 4" not in html
