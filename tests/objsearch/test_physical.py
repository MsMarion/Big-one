"""Always-a-model guarantees (size ladder, photo box, LDraw geometry), physics, datasheet research, export."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from objsearch import acquire, db, docs, export, fallback, models, physical, track
from objsearch.llm.fake_backend import FakeLLM


def _new(conn, oid, **kw):
    db.create_object(conn, {"name": kw.pop("name", oid), "origin": "found", **kw}, oid=oid)
    return db.get_object(conn, oid)


def test_size_ladder(conn):
    _new(conn, "known-thing", category="module", dims_mm=[10, 20, 3])
    assert physical.ensure_dims(conn, "known-thing") == ([10, 20, 3], "catalog")
    _new(conn, "mystery-cam", category="camera")
    with track.run(conn, "t"):
        dims, src = physical.ensure_dims(conn, "mystery-cam")  # no backend: typical size for the category
    assert src == "category_default" and dims == physical.CATEGORY["camera"]["dims"]
    _new(conn, "mystery-drive", category="storage")
    llm = FakeLLM([{"final": {"dims_mm": [38.5, 12.2, 4.6], "basis": "SanDisk Cruzer Fit spec"}}])
    with track.run(conn, "t"):
        dims, src = physical.ensure_dims(conn, "mystery-drive", llm)
    o = db.get_object(conn, "mystery-drive")
    assert (dims, src) == ([38.5, 12.2, 4.6], "photo_estimate") and o["dims_source"] == "photo_estimate"


def test_physics_researched_values_override_estimates(conn):
    o = _new(conn, "esp-x", category="module", dims_mm=[25, 18, 3.5])
    est = physical.physics_for(o)
    assert 0.5 < est["mass_g"] < 2 and est["source"] == "category estimate"
    db.update_object(conn, "esp-x", {"physics": {"mass_g": 4.2, "mass_source": "datasheet", "static_friction": 0.7}})
    got = physical.physics_for(db.get_object(conn, "esp-x"))
    assert got["mass_g"] == 4.2 and got["static_friction"] == 0.7 and got["mass_source"] == "datasheet"


def test_photo_box_has_photo_on_top_and_right_size(tmp_path):
    import trimesh
    photo = tmp_path / "p.png"
    Image.new("RGB", (400, 300), (20, 90, 200)).save(photo)
    out = fallback.photo_box([60, 40, 10], str(tmp_path / "box.glb"), str(photo))
    m = trimesh.load(out, force="mesh")
    assert sorted(np.round(m.extents, 3)) == [10, 40, 60] and np.isclose(m.extents[1], 10)  # glTF: height on Y
    tex = m.visual.material.baseColorTexture
    assert tex is not None and np.asarray(tex)[5, 5].tolist() == [20, 90, 200]  # photo pixel in the atlas
    plain = trimesh.load(fallback.photo_box([5, 5, 5], str(tmp_path / "plain.glb")), force="mesh")
    assert np.allclose(plain.extents, 5)


def _ldraw_lib(root):
    os.makedirs(root / "parts" / "s")
    os.makedirs(root / "p")
    (root / "LDConfig.ldr").write_text("0 !COLOUR Red  CODE   4   VALUE #B40000   EDGE #333333\n"
                                       "0 !COLOUR Blue CODE   1   VALUE #1E5AA8   EDGE #333333\n")
    (root / "p" / "quad.dat").write_text("0 BFC CERTIFY CCW\n4 16 0 0 0 1 0 0 1 0 1 0 0 1\n")
    (root / "parts" / "s" / "sub.dat").write_text("0 BFC CERTIFY CCW\n0 BFC INVERTNEXT\n"
                                                  "1 16 0 0 0 1 0 0 0 1 0 0 0 1 quad.dat\n")
    (root / "parts" / "9999.dat").write_text("0 Test part\n0 BFC CERTIFY CCW\n1 16 0 0 0 1 0 0 0 1 0 0 0 1 quad.dat\n"
                                             "1 1 0 -10 0 1 0 0 0 1 0 0 0 1 s\\sub.dat\n3 16 0 0 0 0 0 1 1 0 0\n")


def test_ldraw_flattens_subfiles_colors_and_bfc(tmp_path):
    from objsearch.sources import ldraw
    _ldraw_lib(tmp_path / "lib")
    lib = ldraw._Lib(str(tmp_path / "lib"), client=object())  # no network: everything is cached locally
    tris, codes = ldraw.triangles("9999", lib, color=4)
    assert len(tris) == 2 + 2 + 1 and sorted(codes.tolist()) == [1, 1, 4, 4, 4]  # 16 -> main / sub color
    n = lambda t: np.cross(t[1] - t[0], t[2] - t[0])[1]  # noqa: E731
    assert n(tris[0]) * n(tris[2]) < 0  # INVERTNEXT flipped the sub-file's copy of the quad
    assert ldraw.color_code(["blue"], lib) == 1 and ldraw.color_code(["purple"], lib) == 4
    out = ldraw.part_glb("9999", str(tmp_path / "p.glb"), ["red"], root=str(tmp_path / "lib"), client=object())
    assert out["triangles"] == 5 and os.path.exists(out["path"]) and out["color_name"] == "red"


def test_acquire_gives_sizeless_objects_a_size_then_a_placeholder(conn, monkeypatch):
    built = []
    monkeypatch.setattr(models, "add_model", lambda c, oid, src, **kw: built.append((oid, src, kw.get("license")))
                        or {"run_id": "r", "flags": []})
    for n in ("_ldraw", "_kicad", "_open_lib", "_github"):
        monkeypatch.setattr(acquire, n, lambda o: None)
    _new(conn, "odd-gadget", category="sensor")
    out = acquire.acquire(conn, "odd-gadget", use_agent=False)
    o = db.get_object(conn, "odd-gadget")
    assert out["how"] == "placeholder" and o["dims_source"] == "category_default" and o["dims_mm"]
    assert built[0][1].endswith("odd-gadget.glb") and built[0][2] == "generated"


def test_docs_research_grabs_pdf_and_sets_physics(conn, monkeypatch):
    from objsearch.sources import web
    _new(conn, "oled", name="SSD1306 OLED", category="sensor", dims_mm=[27, 27, 4])
    pdf = b"%PDF-1.4\n/Title (SSD1306 Datasheet)\n/Type /Page\n/Type /Page\n%%EOF"
    monkeypatch.setattr(web, "download", lambda url, **kw: (url, pdf if url.endswith(".pdf") else b"<html>", "x"))
    llm = FakeLLM([
        {"tool_calls": [{"name": "grab_datasheet", "args": {"url": "https://shop.test/oled.html"}}]},
        {"tool_calls": [{"name": "grab_datasheet", "args": {"url": "https://cdn.test/SSD1306.pdf"}}]},
        {"final": {"datasheet_url": "https://cdn.test/SSD1306.pdf", "mass_g": 3.1, "mass_source": "vendor page",
                   "materials": [{"part": "PCB", "material": "FR4"}, {"part": "panel", "material": "glass"}],
                   "static_friction": 4.0, "restitution": 0.1, "fragile": "glass panel, flex cable"}}])
    res = docs.research(conn, "oled", backend=llm)
    ev = [e for e in db.get_object_full(conn, "oled")["evidence"] if e["kind"] == "datasheet"]
    assert len(ev) == 1 and ev[0]["title"] == "SSD1306 Datasheet" and ev[0]["local_path"].endswith("SSD1306.pdf")
    assert os.path.exists(os.path.join(os.environ["OBJSEARCH_DATA"], ev[0]["local_path"]))
    ph = db.get_object(conn, "oled")["physics"]
    assert ph["mass_g"] == 3.1 and ph["static_friction"] == 1.5 and ph["dynamic_friction"] == 1.2  # clamped
    assert res["datasheet"] and docs.research(conn, "oled", backend=llm) == {"skipped": "already researched"}
    tools = [r[0] for r in conn.execute("SELECT tool FROM tool_calls ORDER BY id")]
    assert tools == ["grab_datasheet", "grab_datasheet"]  # the HTML page was refused, the PDF kept


def test_schema_splitter_refuses_hidden_statements():
    with pytest.raises(ValueError):
        db._split("ALTER TABLE a ADD COLUMN b TEXT;  -- trailing comment\n")
    assert db._split("-- note\nALTER TABLE a ADD COLUMN b TEXT;\n") == ["ALTER TABLE a ADD COLUMN b TEXT;"]


def test_diameter():
    pts = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0], [1, 1, 1]], float)
    assert export._diameter(pts) == 5.0
    big = np.random.default_rng(1).uniform(-1, 1, size=(9000, 3))
    big[0], big[1] = [-10, 0, 0], [10, 0, 0]
    assert export._diameter(big) == 20.0


@pytest.mark.bpy
def test_box_to_sim_usd_and_export(seeded, tmp_path, monkeypatch):
    """Real Blender + pxr: placeholder box -> model -> sim.usda (physics, labels) -> self-contained export."""
    monkeypatch.setenv("OBJSEARCH_SIM", "1")
    out = acquire.acquire(seeded, "esp-01", use_agent=False)
    assert out["result"]["sim"]["colliders"] >= 1
    res = export.export(seeded, str(tmp_path / "exp"), ids=["esp-01"])
    d = tmp_path / "exp" / "objects" / "esp-01"
    info = json.loads((d / "info.json").read_text())
    assert res["classes"] == ["esp-01"] and (d / "mesh" / "mesh.obj").exists() and (d / "usd" / "sim.usda").exists()
    assert abs(info["size_z"] - db.get_object(seeded, "esp-01")["dims_mm"][2]) < 0.5  # OBJ is Z-up like the USD


def test_placeholder_uses_run_photo_cropped_to_the_object(conn, tmp_path, monkeypatch):
    import os
    from objsearch import config
    _new(conn, "plug", name="barrel plug", category="power", dims_mm=[38, 14, 13])
    rel = "runs/r1/inputs/photo_1.jpg"
    os.makedirs(os.path.join(config.data_dir(), "runs/r1/inputs"))
    im = Image.new("RGB", (1000, 500), (120, 80, 40))  # brown desk, black object in the right half
    im.paste((10, 10, 10), (600, 100, 900, 300))
    im.save(os.path.join(config.data_dir(), rel))
    conn.execute("INSERT INTO photos (object_id, path, kind, source, created_at) VALUES ('plug', ?, 'crop', 'run', ?)",
                 (rel, db.now()))
    assert physical.best_photo(conn, "plug").endswith("photo_1.jpg")  # identify-run photo, not yet confirmed
    llm = FakeLLM([{"final": {"found": True, "box_2d": [200, 600, 600, 900]}}])  # [ymin, xmin, ymax, xmax] /1000
    with track.run(conn, "t"):
        cand = acquire._placeholder(conn, db.get_object(conn, "plug"), llm)
    crop = Image.open(os.path.join(config.data_dir(), "placeholders", "plug_crop.jpg"))
    assert abs(crop.size[0] - 380) <= 1 and abs(crop.size[1] - 240) <= 1  # x 0.56-0.94, y 0.16-0.64 of the photo: the box plus 4% padding
    assert max(np.asarray(crop.convert("RGB"))[120, 190]) < 40  # the (dark) object fills the middle
    assert "with your photo on top" in cand["flag"]


@pytest.mark.bpy
def test_recolor_matches_parts_by_size(tmp_path):
    """Real Blender: parts are found by their measured size (CAD exports often share one name), others untouched."""
    import trimesh
    from objsearch.blender import interface
    big = trimesh.creation.box(extents=(0.062, 0.0012, 0.037))           # 'PCB', metres, glTF y-up
    small = trimesh.creation.box(extents=(0.0096, 0.0016, 0.0203))
    small.apply_translation((0, 0.003, 0))
    sc = trimesh.Scene()
    sc.add_geometry(big, geom_name="part")
    sc.add_geometry(small, geom_name="part")
    sc.export(tmp_path / "two.glb")
    res = interface.normalize(str(tmp_path / "two.glb"), str(tmp_path / "out"), [62, 37, 4.6], name="t",
                              render=False, recolor={"pcb": {"size": [62, 37, 1.2], "hex": "#1C4E9C"},
                                                     "nope": {"size": [5, 5, 5], "hex": "#FF0000"}})
    log = (tmp_path / "out" / "blender.log").read_text()
    assert "'pcb': 1" in log and "'nope': 0" in log
    assert any("recolor: parts not found ['nope']" in f for f in res["flags"])
