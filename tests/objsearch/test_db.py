import sqlite3

import pytest
import yaml

from objsearch import db
from conftest import SEED


def test_migrate_is_idempotent(data_dir):
    c1 = db.connect()
    assert c1.execute("PRAGMA user_version").fetchone()[0] == len(db.SCHEMA)
    c1.close()
    c2 = db.connect()  # reconnect must not re-run migrations
    tables = {r[0] for r in c2.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"objects", "photos", "assets", "runs", "steps", "llm_calls", "tool_calls", "quota_usage",
            "object_events", "evals", "sightings", "candidates"} <= tables


def test_create_update_writes_events(conn):
    oid = db.create_object(conn, {"name": "ESP32 thing", "dims_mm": [25.5, 18, 3.1], "aliases": ["wroom"]})
    assert oid == "esp32-thing"
    assert db.create_object(conn, {"name": "ESP32 thing"}) == "esp32-thing-2"
    o = db.get_object(conn, oid)
    assert o["origin"] == "given" and o["owned"] == 1 and o["dims_mm"] == [25.5, 18, 3.1]
    db.update_object(conn, oid, {"status": "identified", "aliases": ["wroom"]}, actor="agent", reason="x")
    ev = db.rows(conn.execute("SELECT field, old, new, actor FROM object_events WHERE object_id=? ORDER BY id", (oid,)))
    assert [e["field"] for e in ev] == ["*created*", "status"]  # unchanged aliases -> no event
    assert ev[1] == {"field": "status", "old": "draft", "new": "identified", "actor": "agent"}


def test_constraints(conn):
    oid = db.create_object(conn, {"name": "x"})
    with pytest.raises(sqlite3.IntegrityError):
        db.update_object(conn, oid, {"status": "bogus"})
    with pytest.raises(ValueError):
        db.update_object(conn, oid, {"id": "nope"})
    with pytest.raises(ValueError):
        db.create_object(conn, {"category": "board"})


def test_fts_follows_updates_and_is_injection_safe(conn):
    oid = db.create_object(conn, {"name": "Old name", "mpn": "ESP-WROOM-32", "aliases": ["devkit"]})
    assert [h["id"] for h in db.search(conn, "wroom")] == [oid]
    assert [h["id"] for h in db.search(conn, "devk")] == [oid]  # prefix match on alias
    db.update_object(conn, oid, {"name": "Shiny new"})
    assert db.search(conn, "old") == [] and db.search(conn, "shiny")
    for nasty in ['ESP"32 OR -x', "NEAR(a b)", "*", "a:b", "'; DROP TABLE objects; --"]:
        db.search(conn, nasty)  # must not raise
        db.list_objects(conn, q=nasty)


def test_seed_import_and_filters(seeded):
    s = db.summary(seeded)
    assert s["all"] == s["seed"] >= 50 and s["owned"] == 0 and s["no_photo"] == s["all"]
    assert db.import_seed(seeded, SEED)["skipped"] == s["all"]  # second import skips existing ids
    lego = db.list_objects(seeded, category="lego")
    assert lego and all(o["category"] == "lego" for o in lego)
    assert [o["id"] for o in db.list_objects(seeded, q="ESP-01")][0] == "esp-01"
    db.update_object(seeded, "esp-01", {"owned": True, "origin": "given"})
    assert [o["id"] for o in db.list_objects(seeded, flag="given")] == ["esp-01"]
    with pytest.raises(ValueError):
        db.list_objects(seeded, flag="nope")


def test_seed_roundtrip_identical(seeded, tmp_path, monkeypatch):
    db.add_photo(seeded, "lego-3001", "photos/lego-3001/a.jpg")
    db.add_photo(seeded, "lego-3001", "photos/lego-3001/b.jpg", view="bottom")
    db.add_asset(seeded, "lego-3001", "usd", "assets/lego-3001/asset.usd", license="CC-BY-2.0",
                 source_url="https://library.ldraw.org/", dims_mm=[31.8, 15.8, 11.4], tri_count=1200)
    db.add_evidence(seeded, "lego-3001", "https://example.test/3001", title="3001", kind="product", trust="user")
    p1 = tmp_path / "a.yaml"
    t1 = db.export_seed(seeded, str(p1))
    monkeypatch.setenv("OBJSEARCH_DATA", str(tmp_path / "other"))
    c2 = db.connect()
    db.import_seed(c2, str(p1))
    assert db.export_seed(c2) == t1
    e = next(o for o in yaml.safe_load(t1)["objects"] if o["id"] == "lego-3001")
    assert e["primary_photo"] == "photos/lego-3001/a.jpg" and len(e["photos"]) == 2 and e["assets"][0]["tri_count"] == 1200


def test_get_object_full_and_runs(seeded):
    o = db.get_object_full(seeded, "esp-01")
    assert o["photos"] == [] and o["events"][0]["field"] == "*created*"
    assert db.get_object_full(seeded, "missing") is None
    assert db.list_runs(seeded) == [] and db.get_run(seeded, "nope") is None


def test_candidates_dedupe_by_source(seeded):
    a = db.add_candidate(seeded, "esp-01", source_url="https://example.test/m", rank=2, scores={"score": 0.5})
    b = db.add_candidate(seeded, "esp-01", source_url="https://example.test/m", rank=1, scores={"score": 0.8})
    c = db.add_candidate(seeded, "esp-01", source_url="https://example.test/other", rank=3)
    rows = db.get_object_full(seeded, "esp-01")["candidates"]
    assert a == b != c and len(rows) == 2 and rows[0]["scores"] == {"score": 0.8}


def test_reset_inventory_keeps_knowledge(seeded):
    from objsearch import track
    db.update_object(seeded, "esp-01", {"owned": True, "origin": "given", "model_hints": [{"url": "https://m.test/a.step"}]})
    mine = db.create_object(seeded, {"name": "My weird board"})
    db.add_photo(seeded, "esp-01", "photos/esp-01/a.jpg")
    db.add_ref_image(seeded, "esp-01", path="refs/esp-01/x.jpg", provider_id="x", rank=1, chosen=1)
    with track.run(seeded, "test"):
        db.add_asset(seeded, "esp-01", "glb", "model_cache/k/model.glb")
    seeded.execute("INSERT INTO parts3d (key, kind, status, created_at, updated_at) VALUES ('k','kicad_model','real','t','t')")
    n = db.reset_inventory(seeded)
    o = db.get_object_full(seeded, "esp-01")
    assert (o["owned"], o["origin"], o["photos"], o["assets"]) == (0, "seed", [], [])
    assert o["model_hints"] == [{"url": "https://m.test/a.step"}] and len(o["ref_images"]) == 1   # knowledge kept
    assert db.get_object(seeded, mine) is None and n["objects_removed"] == 1
    assert seeded.execute("SELECT COUNT(*) FROM parts3d").fetchone()[0] == 1
    assert db.summary(seeded)["owned"] == 0 and db.list_runs(seeded) == []
