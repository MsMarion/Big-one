import math
import os

import pytest
from fastapi.testclient import TestClient

from objsearch import config, db, kicad, models, parts, track

FOOTPRINT = """(footprint "TEST_JACK" (layer "F.Cu")
  (fp_line (start -1 -3) (end 12 -3) (layer "F.Fab"))
  (fp_line (start 12 -3) (end 12 3) (layer "F.Fab"))
  (fp_line (start 12 3) (end -1 3) (layer "F.Fab"))
  (fp_line (start -1 3) (end -1 -3) (layer "F.Fab"))
  (pad "1" thru_hole circle (at 0 -1.5) (size 1.6 1.6) (drill 1) (layers "*.Cu"))
  (pad "2" thru_hole circle (at 0 1.5) (size 1.6 1.6) (drill 1) (layers "*.Cu"))
)
"""


def _vendor_step(path):
    """A 'vendor' part drawn in its own frame: body 13 (long, along Y) x 6 x 4 (tall, along Z... rotated so the
    long axis is Y and up is -X), with two pins at the y=-6.5 end poking out below the body."""
    g = parts._gmsh()
    occ = g.model.occ
    occ.addBox(-4, -6.5, -3, 4, 13, 6)          # body: x -4..0 (height 4 along -X), y -6.5..6.5, z -3..3
    occ.addBox(0, -6.4, -1.8, 3, 0.6, 0.6)      # pin at z=-1.5 near the y=-6.5 end, sticking out to +X (= down)
    occ.addBox(0, -6.4, 1.2, 3, 0.6, 0.6)       # pin at z=+1.5
    occ.synchronize()
    g.write(str(path))
    return str(path)


def test_key_for_and_upsert_never_downgrades(conn):
    assert parts.key_for("${KICAD10_3DMODEL_DIR}/Connector_USB.3dshapes/X.wrl") == "Connector_USB.3dshapes/X"
    parts.upsert(conn, "L.3dshapes/A", "kicad_model", "approximate", "obj-1", path="parts3d/approx/a.step")
    p = parts.upsert(conn, "L.3dshapes/A", "kicad_model", "real", "obj-2", path="parts3d/real/a.step", license="CC0")
    assert p["status"] == "real" and p["seen_in"] == ["obj-1", "obj-2"]
    p = parts.upsert(conn, "L.3dshapes/A", "kicad_model", "approximate", "obj-3", path="x")
    assert p["status"] == "real" and p["path"] == "parts3d/real/a.step" and "obj-3" in p["seen_in"]
    assert parts.affected_objects(conn) == ["obj-1", "obj-2", "obj-3"]


def test_auto_align_puts_leads_on_pads(data_dir, tmp_path):
    fp = tmp_path / "TEST_JACK.kicad_mod"
    fp.write_text(FOOTPRINT)
    src = _vendor_step(tmp_path / "vendor.step")
    cands = parts.auto_align(src, str(fp), height=4, all_candidates=True)
    assert len(cands) >= 2  # size alone can't tell front/back (or up/down)
    best, scores = parts.orient_by_pads(src, cands, str(fp))
    ok = sorted(x for x in scores if x != float("inf"))
    assert ok[0] < 1.0 and ok[1] > 3.0  # only one orientation lands the pins on the pads
    pins = parts._pin_points(src, best)
    pads = [(0, 1.5), (0, -1.5)]  # 3D y = -footprint y
    assert len(pins) == 2 and all(min(math.dist(p, q) for q in pads) < 1.0 for p in pins)
    dst = tmp_path / "aligned.step"
    dims = parts.transform_step(src, str(dst), best)
    assert abs(sorted(dims)[-1] - 13) < 0.2  # long axis preserved


def test_unmapped_physical_part_gets_flagged_approximation(conn, monkeypatch):
    from test_kicad import BOARD
    monkeypatch.setattr(kicad, "index_ready", lambda: False)
    with track.run(conn, "test"):
        text, rep = kicad.attach_models(kicad.sanitize(BOARD)[0], mapping={}, conn=conn, object_id="esp-01")
    assert rep["approximate_unmapped"] == {"1X03": ["J1"], "WEIRD": ["?"]} and rep["placed"] == 2
    p = parts.get(conn, "eagle:1X03")
    assert p["status"] == "approximate" and p["seen_in"] == ["esp-01"] and p["transform"]["footprint"].endswith(".kicad_mod")
    assert os.path.exists(os.path.join(config.data_dir(), p["path"]))
    flags = models.part_flags(rep)
    assert any("J1 (1X03)" in f and "approximate" in f for f in flags)
    # the approximation is reused (not regenerated) on the next build
    with track.run(conn, "test"):
        _, rep2 = kicad.attach_models(kicad.sanitize(BOARD)[0], mapping={}, conn=conn, object_id="lego-3001")
    assert parts.get(conn, "eagle:1X03")["seen_in"] == ["esp-01", "lego-3001"] and rep2["placed"] == 2


def test_fit_unlabeled_ignores_names():
    lib = [(0, 0), (6.3, 0), (0, 4.5), (6.3, 4.5)]
    board = [(10 + y, 5 - x) for x, y in lib] + [(10 + 2.25, 5)]  # rotated 90 + an extra centre pad
    f = kicad.fit_unlabeled(lib, board)
    assert f and f[3] == 4 and f[2] < 1e-6
    assert kicad.fit_unlabeled([(0, 0), (9, 0)], board) is None


def test_parts_api_add_real(seeded, tmp_path):
    from dashboard.app import app
    client = TestClient(app)
    fp = os.path.join(config.data_dir(), "parts3d", "footprints", "eagle__TEST_JACK.kicad_mod")
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    open(fp, "w").write(FOOTPRINT)
    parts.upsert(seeded, "eagle:TEST_JACK", "footprint", "approximate", "esp-01",
                 transform={"footprint": "parts3d/footprints/eagle__TEST_JACK.kicad_mod"})
    d = client.get("/api/parts").json()
    assert d["counts"]["approximate"] == 1 and d["parts"][0]["used_by"][0]["name"].startswith("ESP-01")
    step = _vendor_step(tmp_path / "v.step")
    with open(step, "rb") as f:
        bad = client.post("/api/parts/real", data={"key": "eagle:TEST_JACK"}, files={"file": ("v.step", f.read())})
    assert bad.status_code == 400  # license required
    with open(step, "rb") as f:
        r = client.post("/api/parts/real", data={"key": "eagle:TEST_JACK", "license": "CC0", "rebuild": "false"},
                        files={"file": ("v.step", f.read(), "application/octet-stream")})
    assert r.status_code == 202
    import time
    for _ in range(200):
        if db.get_run(seeded, r.json()["run_id"])["status"] != "running":
            break
        time.sleep(0.05)
    p = parts.get(seeded, "eagle:TEST_JACK")
    assert p["status"] == "real" and "lead-to-pad" in p["notes"] and p["license"] == "CC0"
    assert client.post("/api/parts/real", data={"key": "nope", "license": "x", "url": "https://x.test/a.step"}).status_code == 404


def test_static_pages_win_over_object_route(seeded):
    from dashboard.app import app
    client = TestClient(app)
    for path, marker in (("/objects/parts", "Parts library"), ("/objects/identify", "Identify"),
                         ("/objects/check", "Check my collection"), ("/objects/runs", "Runs")):
        assert marker in client.get(path).text, path


def test_open_library_candidates_rank_variants(seeded, monkeypatch):
    monkeypatch.setattr(parts, "library_index", lambda name="freecad", refresh=False: [
        "Electronics Parts/Connectors/USB-connectors/smd/USB-micro-B.step",
        "Electronics Parts/Connectors/USB-connectors/USB-2_0-type-B-jack-PCB.step",
        "Mechanical Parts/Gears/spur-gear.step"])
    c = parts.search_open_libraries("Connector_USB.3dshapes/USB_B_OST_USB-B1HSxx_Horizontal")
    assert [x["path"].split("/")[-1] for x in c] == ["USB-2_0-type-B-jack-PCB.step", "USB-micro-B.step"]
    assert c[0]["license"].startswith("CC-BY-3.0") and c[0]["url"].startswith("https://raw.githubusercontent.com/")
    assert parts.search_open_libraries("eagle:TS42") == []
    from dashboard.app import app
    client = TestClient(app)
    parts.upsert(seeded, "Connector_USB.3dshapes/USB_B_OST_USB-B1HSxx_Horizontal", "kicad_model", "approximate")
    r = client.get("/api/parts/candidates", params={"key": "Connector_USB.3dshapes/USB_B_OST_USB-B1HSxx_Horizontal"})
    assert r.json()["candidates"][0]["path"].endswith("USB-2_0-type-B-jack-PCB.step")
    assert client.get("/api/parts/candidates", params={"key": "nope"}).status_code == 404
