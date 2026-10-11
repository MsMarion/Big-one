import io
import os

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image

from objsearch import config, db, refimages, track
from objsearch.sources import openverse


def _jpeg(color=(30, 90, 200)):
    buf = io.BytesIO()
    Image.new("RGB", (1200, 900), color).save(buf, "JPEG")
    return buf.getvalue()


def _result(i, title, tags=()):
    return {"id": f"{i:08d}-0000-0000-0000-000000000000", "title": title, "tags": [{"name": t} for t in tags],
            "creator": f"creator {i}", "creator_url": "https://example.test/u", "license": "by", "license_version": "2.0",
            "license_url": "https://creativecommons.org/licenses/by/2.0/", "source": "flickr",
            "foreign_landing_url": f"https://example.test/p/{i}", "url": f"https://example.test/i/{i}.jpg",
            "attribution": f'"{title}" by creator {i} is licensed under CC BY 2.0.', "mature": False}


RESULTS = [_result(1, "ESP-01 with adapter", ["esp8266"]), _result(2, "Guns n LEGOs"),
           _result(3, "esp01 module closeup", ["esp8266", "wifi"]), _result(4, "ESP-01 ESP8266 board"),
           _result(5, "ESP8266 ESP-01 pinout photo")]


@pytest.fixture
def ov():
    with respx.mock(assert_all_called=False) as m:
        search = m.get(openverse.API).mock(return_value=httpx.Response(200, json={"results": RESULTS}))
        for r in RESULTS:
            base = f"{openverse.API}{r['id']}/thumb/"
            m.get(base, params={"full_size": "true"}).mock(return_value=httpx.Response(
                424 if r["title"].startswith("esp01") else 200, content=_jpeg(), headers={"content-type": "image/jpeg"}))
            m.get(base).mock(return_value=httpx.Response(200, content=_jpeg((0, 200, 0)),
                                                         headers={"content-type": "image/jpeg"}))
        yield search


def test_fetch_ranks_filters_and_credits(seeded, ov):
    with track.run(seeded, "test"):
        assert refimages.fetch_for_object(seeded, "esp-01", n=3, pace=False) == 3
    refs = db.get_object_full(seeded, "esp-01")["ref_images"]
    assert len(refs) == 3 and refs[0]["chosen"] == 1 and sum(r["chosen"] for r in refs) == 1
    assert all("LEGO" not in r["title"] for r in refs)  # irrelevant result dropped
    assert all(r["license"] == "by" and r["attribution"] and r["landing_url"] for r in refs)
    for r in refs:
        full = os.path.join(config.data_dir(), r["path"])
        assert max(Image.open(full).size) <= 900 and os.path.exists(full.replace(".jpg", ".thumb.jpg"))
    # list view exposes the chosen image for the cards
    o = db.list_objects(seeded, q="ESP-01")[0]
    assert o["ref_path"] == refs[0]["path"] and "CC BY" in o["ref_attribution"]
    # second run: skipped without force; with a query it only adds new images (no duplicates)
    with track.run(seeded, "test"):
        assert refimages.fetch_for_object(seeded, "esp-01", pace=False) == 0
        assert refimages.fetch_for_object(seeded, "esp-01", n=3, query="ESP-01", pace=False) == 1
    ids = [r["provider_id"] for r in db.get_object_full(seeded, "esp-01")["ref_images"]]
    assert len(ids) == len(set(ids)) == 4


def test_424_falls_back_to_small_thumb(seeded, ov):
    with track.run(seeded, "test") as r:
        refimages.fetch_for_object(seeded, "esp-01", n=4, pace=False)
    http = db.get_run(seeded, r.id)["http_requests"]
    assert any(h["status"] == 424 for h in http) and all(h["status"] in (200, 424) for h in http)


def test_no_match_and_quota_skip(seeded, ov, monkeypatch):
    ov.mock(return_value=httpx.Response(200, json={"results": [_result(9, "unrelated sunset")]}))
    with track.run(seeded, "test") as r:
        assert refimages.fetch_for_object(seeded, "hc-sr04", pace=False) == 0
        track.bump_quota(seeded, "openverse", "images", "openverse_search", openverse.SEARCH_LIMIT_PER_DAY)
        assert refimages.fetch_for_object(seeded, "dht11", pace=False) == 0
    steps = db.get_run(seeded, r.id)["steps"]
    assert [s["status"] for s in steps] == ["skipped", "skipped"]
    assert "quota" in steps[1]["outputs"]["skip_reason"]


def test_fetch_all_tracks_one_run(seeded, ov, monkeypatch):
    monkeypatch.setattr(openverse, "_pace", lambda *a: None)
    out = refimages.fetch_all(seeded, ids=["esp-01", "esp-12e"], n=2)
    assert out["objects"] == 2 and out["failed"] == 0
    assert db.get_run(seeded, out["run_id"])["kind"] == "refimages"


def test_migration_v1_to_v2_keeps_data(data_dir, monkeypatch):
    full = db.SCHEMA
    monkeypatch.setattr(db, "SCHEMA", full[:1])
    c = db.connect()
    db.create_object(c, {"name": "Old object", "dims_mm": [1, 2, 3]})
    assert c.execute("PRAGMA user_version").fetchone()[0] == 1
    c.close()
    monkeypatch.setattr(db, "SCHEMA", full)  # not undo(): that would also drop OBJSEARCH_DATA
    c = db.connect()
    assert str(data_dir) in c.execute("PRAGMA database_list").fetchone()[2]
    assert c.execute("PRAGMA user_version").fetchone()[0] == len(db.SCHEMA) >= 2
    o = db.get_object(c, "old-object")
    assert o["dims_mm"] == [1, 2, 3] and o["ref_query"] is None and o["collection_checked_at"] is None
    assert db.summary(c)["unchecked"] == 1


def test_check_and_ref_endpoints(seeded, ov):
    from dashboard.app import app
    client = TestClient(app)
    with track.run(seeded, "test"):
        refimages.fetch_for_object(seeded, "esp-01", n=3, pace=False)
        refimages.fetch_for_object(seeded, "esp-12e", n=1, query="ESP8266", pace=False)
    refs = client.get("/api/objects/esp-01").json()["ref_images"]
    other = client.get("/api/objects/esp-12e").json()["ref_images"][0]
    # choosing another object's image is refused and changes nothing
    assert client.post(f"/api/objects/esp-01/refimages/{other['id']}/choose").status_code == 404
    assert client.get("/api/objects/esp-01").json()["ref_images"][0]["chosen"] == 1
    o = client.post(f"/api/objects/esp-01/refimages/{refs[2]['id']}/choose").json()
    assert next(r for r in o["ref_images"] if r["chosen"])["id"] == refs[2]["id"]
    o = client.delete(f"/api/objects/esp-01/refimages/{refs[2]['id']}").json()
    assert len(o["ref_images"]) == 2 and sum(r["chosen"] for r in o["ref_images"]) == 1
    assert not os.path.exists(os.path.join(config.data_dir(), refs[2]["path"]))
    # collection check
    assert client.get("/objects/check").status_code == 200
    assert client.post("/api/objects/esp-01/check", json={"have": "yes"}).status_code == 400
    o = client.post("/api/objects/esp-01/check", json={"have": True}).json()
    assert (o["owned"], o["origin"], o["include_in_search"]) == (1, "given", 1) and o["collection_checked_at"]
    o = client.post("/api/objects/lego-3001/check", json={"have": False}).json()
    assert (o["owned"], o["origin"], o["include_in_search"]) == (0, "seed", 0) and o["collection_checked_at"]
    s = client.get("/api/objects").json()["summary"]
    assert s["unchecked"] == s["all"] - 2 and s["owned"] == 1


def _png_rgba():
    buf = io.BytesIO()
    Image.new("RGBA", (300, 224), (255, 220, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


def test_lego_uses_ldraw_render(seeded):
    from objsearch.sources import ldraw
    page = ('<img src="https://library.ldraw.org/media/parts/3436/conversions/3001-thumb.png?v=1">'
            '<img src="https://library.ldraw.org/media/parts/3437/conversions/30010-thumb.png?v=1">')
    dat = "0 Brick  2 x  4\n0 Name: 3001.dat\n0 Author: James Jessiman\n0 !LICENSE Licensed under CC BY 4.0 : see CAreadme.txt\n"
    with respx.mock() as m:
        m.get(f"{ldraw.BASE}/parts/list").mock(return_value=httpx.Response(200, text=page))
        m.get(f"{ldraw.BASE}/library/official/parts/3001.dat").mock(return_value=httpx.Response(200, text=dat))
        m.get(f"{ldraw.BASE}/media/parts/3436/3001.png").mock(
            return_value=httpx.Response(200, content=_png_rgba(), headers={"content-type": "image/png"}))
        with track.run(seeded, "test"):
            assert refimages.fetch_for_object(seeded, "lego-3001", pace=False) == 1
            assert refimages.fetch_for_object(seeded, "lego-3001", force=True, pace=False) == 0  # no duplicate
    r = db.get_object_full(seeded, "lego-3001")["ref_images"]
    assert len(r) == 1 and r[0]["chosen"] == 1 and r[0]["provider"] == "ldraw.org"
    assert r[0]["creator"] == "James Jessiman" and r[0]["license_version"] == "4.0"
    assert "LDraw.org Parts Library" in r[0]["attribution"] and r[0]["title"] == "LDraw 3001: Brick 2 x 4"
    im = Image.open(os.path.join(config.data_dir(), r[0]["path"]))
    assert im.mode == "RGB" and im.getpixel((5, 5)) == (255, 255, 255)  # transparency -> white
