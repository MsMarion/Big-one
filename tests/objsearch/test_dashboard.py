import io
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from objsearch import config, db, gallery


@pytest.fixture
def client(seeded):
    from dashboard.app import app
    return TestClient(app)


def _png(color=(200, 30, 30), size=(800, 600)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_pages_and_list(client):
    for p in ("/objects", "/objects/esp-01", "/objects/runs", "/objects/runs/x", "/objects-static/objects.js"):
        assert client.get(p).status_code == 200, p
    assert client.get("/objects-static/app.py").status_code == 404
    d = client.get("/api/objects").json()
    assert d["summary"]["seed"] == len(d["objects"]) >= 50
    assert client.get("/api/objects?flag=bogus").status_code == 400
    lego = client.get("/api/objects", params={"category": "lego", "q": "2 x 4"}).json()["objects"]
    assert {o["id"] for o in lego} >= {"lego-3001"}


def test_create_patch_and_validation(client):
    r = client.post("/api/objects", json={"name": "My ESP32-CAM clone", "dims_mm": [40, 27, 12], "category": "camera"})
    assert r.status_code == 201 and r.json()["origin"] == "given" and r.json()["owned"] == 1
    oid = r.json()["id"]
    assert client.post("/api/objects", json={"category": "x"}).status_code == 400
    assert client.patch(f"/api/objects/{oid}", json={"dims_mm": [1, 2]}).status_code == 400
    assert client.patch(f"/api/objects/{oid}", json={"created_at": "x"}).status_code == 400
    assert client.patch(f"/api/objects/{oid}", json={"status": "bogus"}).status_code == 400
    assert client.patch("/api/objects/nope", json={"name": "x"}).status_code == 404
    o = client.patch(f"/api/objects/{oid}", json={"aliases": [" a ", "", "b"], "notes": "  "}).json()
    assert o["aliases"] == ["a", "b"] and o["notes"] is None
    # owning a seed item converts it to your collection
    s = client.patch("/api/objects/esp-01", json={"owned": True}).json()
    assert (s["origin"], s["owned"], s["include_in_search"]) == ("given", 1, 1)
    assert [e["field"] for e in s["events"][:3]] == ["include_in_search", "origin", "owned"] or \
        {e["field"] for e in s["events"][:3]} == {"include_in_search", "origin", "owned"}


def test_photo_upload_strips_and_thumbs(client):
    r = client.post("/api/objects/lego-3001/photos",
                    files=[("files", ("a.png", _png(), "image/png")), ("files", ("b.txt", b"hello", "text/plain"))])
    assert r.status_code == 200 and len(r.json()["added"]) == 1 and r.json()["errors"]
    o = client.get("/api/objects/lego-3001").json()
    p = o["photos"][0]
    assert o["primary_photo_id"] == p["id"] and p["path"].endswith(".jpg")
    full = os.path.join(config.data_dir(), p["path"])
    assert max(Image.open(full).size) <= 1024 and os.path.exists(full.replace(".jpg", ".thumb.jpg"))
    assert client.get(f"/objdata/{p['path']}").headers["content-type"] == "image/jpeg"
    assert client.post("/api/objects/lego-3001/photos", files=[("files", ("x.jpg", b"notimage", "image/jpeg"))]).status_code == 400
    assert client.delete(f"/api/objects/lego-3001/photos/{p['id']}").status_code == 200
    assert not os.path.exists(full)
    assert client.get("/api/objects/lego-3001").json()["primary_photo_id"] is None


@pytest.mark.parametrize("path", ["../objsearch.db", "objsearch.db", "..%2Fobjsearch.db", "photos/../../x.jpg",
                                  "C:/Windows/win.ini", "/etc/passwd", "runs/x/../../objsearch.db"])
def test_objdata_rejects_escapes(client, path):
    assert client.get(f"/objdata/{path}").status_code in (400, 404)


def test_text_endpoint_only_serves_run_files(client):
    assert client.get("/api/text", params={"path": "objsearch.db"}).status_code == 400
    assert client.get("/api/text", params={"path": "runs/../objsearch.db"}).status_code in (400, 404)


def test_approve_reject_and_draft(client, seeded):
    oid = client.post("/api/objects", json={"name": "Mystery board"}).json()["id"]
    assert client.post(f"/api/objects/{oid}/approve", json={"what": "model"}).status_code == 400
    assert client.post(f"/api/objects/{oid}/approve", json={"what": "identity"}).json()["status"] == "identified"
    cid = db.add_candidate(seeded, oid, source_url="https://example.test/m", license="CC-BY-4.0", rank=1)
    o = client.post(f"/api/objects/{oid}/approve", json={"what": "model", "candidate_id": cid}).json()
    assert o["status"] == "verified" and o["candidates"][0]["chosen"] == 1
    assert client.post(f"/api/objects/{oid}/draft").status_code == 501
    assert client.delete(f"/api/objects/{oid}").status_code == 200
    assert client.get(f"/api/objects/{oid}").json()["status"] == "rejected"
    assert oid not in {o["id"] for o in client.get("/api/objects").json()["objects"]}


def test_demo_endpoint_and_runs_api(client):
    assert client.post("/api/runs/demo", json={"fail_at": "nope"}).status_code == 400
    assert client.post("/api/runs/demo", json={"fail_at": "acquire"}).status_code == 202
    for _ in range(100):
        runs = client.get("/api/runs").json()["runs"]
        if runs and runs[0]["status"] != "running":
            break
        time.sleep(0.05)
    r = client.get(f"/api/runs/{runs[0]['id']}").json()
    assert r["status"] == "partial" and r["steps"] and r["sightings"]
    assert '"msg"' in client.get(f"/api/runs/{r['id']}/log").text
    tr = next(s for s in r["steps"] if s["stage"] == "identify")["outputs"]["transcript"]
    assert "assistant" in client.get("/api/text", params={"path": tr}).text
    assert client.get("/api/quota").json()["grounded_search"]["used"] >= 1
    assert client.get("/api/usage").json()["by_stage"]
    assert client.get("/api/activity").json()["failed_steps"]


def test_gallery_escapes_and_embeds(seeded, data_dir):
    oid = db.create_object(seeded, {"name": "<script>alert(1)</script>"})
    from objsearch import photos
    db.add_photo(seeded, oid, photos.ingest(oid, _png()))
    html = gallery.render(seeded)
    assert "<script>alert" not in html and "&lt;script&gt;" in html and "data:image/jpeg;base64," in html
    assert os.path.exists(gallery.write(seeded))
