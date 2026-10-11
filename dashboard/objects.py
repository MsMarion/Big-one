"""Objects / runs pages and API for the local dashboard, backed by the objsearch SQLite DB
(no SSH involved). Included by dashboard.app.

Pages: /objects, /objects/{id}, /objects/runs, /objects/runs/{run_id}
API:   /api/objects*, /api/runs*, /api/activity, /api/quota, /api/usage, /objdata/{path}
"""
import os
import re
import threading

from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from objsearch import config, db, grasp, photos, track

HERE = os.path.dirname(os.path.abspath(__file__))
router = APIRouter()

SAFE_REL = re.compile(r"^[\w.-]+(/[\w.-]+)*$")
DATA_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
              ".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".obj": "text/plain", ".mtl": "text/plain",
              ".usd": "application/octet-stream", ".usda": "text/plain", ".usdc": "application/octet-stream",
              ".stl": "model/stl", ".step": "application/octet-stream", ".stp": "application/octet-stream",
              ".json": "application/json", ".jsonl": "text/plain", ".txt": "text/plain", ".log": "text/plain",
              ".pdf": "application/pdf"}  # downloaded datasheets (shown in the browser's PDF viewer)
DOWNLOAD_EXT = {".usd", ".usdc", ".step", ".stp", ".stl", ".obj", ".mtl"}
PAGES = {"objects.css", "objects.js"}  # check.html etc. are served by their own routes


_local = threading.local()


def _conn():
    """One connection per worker thread (FastAPI reuses its threadpool), reopened if data_dir changes."""
    path = config.db_path()
    c = getattr(_local, "conn", None)
    if c is None or _local.path != path:
        if c is not None:
            c.close()
        _local.conn, _local.path = db.connect(path), path
    return _local.conn


def _page(name):
    return FileResponse(os.path.join(HERE, name), headers={"Cache-Control": "no-cache"})


def _data_file(rel):
    if not SAFE_REL.match(rel) or ".." in rel.split("/"):
        raise HTTPException(400, "bad path")
    root = os.path.realpath(config.data_dir())
    full = os.path.realpath(os.path.join(root, rel))
    if not full.startswith(root + os.sep) or not os.path.isfile(full):
        raise HTTPException(404, "not found")
    return full


# ---------------------------------------------------------------- pages

@router.get("/objects")
def objects_page():
    return _page("objects.html")


@router.get("/objects/check")
def check_page():
    return _page("check.html")


@router.get("/objects/parts")
def parts_page():
    return _page("parts.html")


@router.get("/objects/identify")
def identify_page():
    return _page("identify.html")


@router.get("/objects/runs")
@router.get("/objects/runs/{run_id}")
def runs_page(run_id: str = ""):
    return _page("runs.html")


@router.get("/objects/{oid}")
def object_page(oid: str):
    return _page("object.html")


@router.get("/objects-static/{name}")
def static(name: str):
    if name not in PAGES:
        raise HTTPException(404)
    return _page(name)


@router.get("/objdata/{rel:path}")
def objdata(rel: str):
    full = _data_file(rel)
    ext = os.path.splitext(full)[1].lower()
    if ext not in DATA_MEDIA:
        raise HTTPException(400, "unsupported file type")
    headers = {"Cache-Control": "max-age=60", "X-Content-Type-Options": "nosniff"}
    if ext in DOWNLOAD_EXT:
        headers["Content-Disposition"] = f'attachment; filename="{os.path.basename(full)}"'
    return FileResponse(full, media_type=DATA_MEDIA[ext], headers=headers)


# ---------------------------------------------------------------- objects API

def _clean_changes(changes):
    """Validate PATCH/POST bodies against the editable fields and their types."""
    out = {}
    for k, v in changes.items():
        if k not in db.EDITABLE:
            raise HTTPException(400, f"field {k!r} is not editable")
        if k == "dims_mm" and v is not None:
            if not (isinstance(v, list) and len(v) == 3 and all(isinstance(x, (int, float)) and x > 0 for x in v)):
                raise HTTPException(400, "dims_mm must be [l, w, h] in mm, all > 0")
            v = [float(x) for x in v]
        if k in ("aliases", "colors") and v is not None:
            if not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                raise HTTPException(400, f"{k} must be a list of strings")
            v = [x.strip() for x in v if x.strip()]
        if k in ("owned", "include_in_search"):
            v = bool(v)
        if k == "confidence" and v is not None and not (isinstance(v, (int, float)) and 0 <= v <= 1):
            raise HTTPException(400, "confidence must be 0..1")
        if isinstance(v, str):
            v = v.strip() or None
        out[k] = v
    if "status" in out and out["status"] not in ("draft", "identified", "model_found", "verified", "rejected"):
        raise HTTPException(400, "bad status")
    if "origin" in out and out["origin"] not in ("given", "found", "seed"):
        raise HTTPException(400, "bad origin")
    if "grasp_pose" in out and out["grasp_pose"] not in (None, "flat"):
        raise HTTPException(400, "grasp_pose must be null (as listed) or 'flat'")
    return out


@router.get("/api/objects")
def list_objects(flag: str | None = None, category: str | None = None, q: str | None = None,
                 status: str | None = None, grip: str | None = None):
    """grip: good | borderline | no | ok (= good + borderline), from objsearch.grasp (SO-ARM101 limits)."""
    conn = _conn()
    if grip and grip not in grasp.FILTERS:
        raise HTTPException(400, f"grip must be one of {sorted(grasp.FILTERS)}")
    try:
        objs = grasp.annotate(db.list_objects(conn, flag=flag, category=category, q=q, status=status))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if grip:
        objs = [o for o in objs if o["grasp"]["verdict"] in grasp.FILTERS[grip]]
    summary = db.summary(conn)
    summary["grip"] = grasp.counts(grasp.annotate(db.list_objects(conn)))
    summary["gripper"] = grasp.limits()
    return {"objects": objs, "summary": summary}


@router.post("/api/objects", status_code=201)
def create_object(body: dict = Body(...)):
    f = _clean_changes(body)
    if not f.get("name"):
        raise HTTPException(400, "name is required")
    f.setdefault("origin", "given")
    f.setdefault("status", "draft")
    conn = _conn()
    oid = db.create_object(conn, f, actor="user", reason="added in dashboard")
    return _full(conn, oid)


def _full(conn, oid):
    o = db.get_object_full(conn, oid)
    if o is not None:
        grasp.annotate([o])
        from objsearch import physical, sim
        o["physics_final"] = physical.physics_for(o)  # researched values over category estimates
        o["sim_usd"] = track.rel(sim.path(oid)) if os.path.exists(sim.path(oid)) else None
        o["docs_running"] = oid in _docs_running
    return o


_docs_running = set()


@router.post("/api/objects/{oid}/docs")
def research_docs(oid: str):
    """Datasheet + mass/materials research in the background (about a minute); the page polls the object."""
    from objsearch import docs
    if db.get_object(_conn(), oid) is None:
        raise HTTPException(404, "no such object")
    if oid in _docs_running:
        return {"started": False, "running": True}
    _docs_running.add(oid)

    def work():
        conn = db.connect()
        try:
            docs.research(conn, oid, force=True)
        except Exception as e:  # recorded in the run log; the page shows the run
            import logging
            logging.getLogger("objsearch.dashboard").warning(f"docs research {oid}: {e}")
        finally:
            _docs_running.discard(oid)
            conn.close()
    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@router.get("/api/objects/{oid}")
def get_object(oid: str):
    o = _full(_conn(), oid)
    if o is None:
        raise HTTPException(404, "no such object")
    return o


@router.patch("/api/objects/{oid}")
def patch_object(oid: str, body: dict = Body(...)):
    conn = _conn()
    old = db.get_object(conn, oid)
    if old is None:
        raise HTTPException(404, "no such object")
    ch = _clean_changes(body)
    if ch.get("owned") and old["origin"] == "seed":  # owning a seed item makes it part of your collection
        ch.setdefault("origin", "given")
        ch.setdefault("include_in_search", True)
    db.update_object(conn, oid, ch, actor="user", reason="edited in dashboard")
    return _full(conn, oid)


@router.delete("/api/objects/{oid}")
def reject_object(oid: str):
    conn = _conn()
    if db.get_object(conn, oid) is None:
        raise HTTPException(404, "no such object")
    db.update_object(conn, oid, {"status": "rejected", "include_in_search": False}, actor="user",
                     reason="rejected in dashboard")
    return {"ok": True}


@router.post("/api/objects/{oid}/photos")
async def upload_photos(oid: str, files: list[UploadFile] = File(...)):
    conn = _conn()
    if db.get_object(conn, oid) is None:
        raise HTTPException(404, "no such object")
    added, errors = [], []
    for f in files[:20]:
        data = await f.read(photos.MAX_UPLOAD + 1)
        try:
            rel = photos.ingest(oid, data)
        except ValueError as e:
            errors.append(f"{f.filename}: {e}")
            continue
        added.append(db.add_photo(conn, oid, rel, kind="ref", source="user", actor="user"))
    if not added and errors:
        raise HTTPException(400, "; ".join(errors))
    return {"added": added, "errors": errors}


@router.post("/api/objects/{oid}/photos/{pid}/primary")
def primary_photo(oid: str, pid: int):
    conn = _conn()
    if not conn.execute("SELECT 1 FROM photos WHERE id=? AND object_id=?", (pid, oid)).fetchone():
        raise HTTPException(404, "no such photo")
    db.update_object(conn, oid, {"primary_photo_id": pid}, actor="user", reason="primary photo")
    return {"ok": True}


@router.delete("/api/objects/{oid}/photos/{pid}")
def delete_photo(oid: str, pid: int):
    conn = _conn()
    p = db.row(conn.execute("SELECT * FROM photos WHERE id=? AND object_id=? AND source='user'", (pid, oid)).fetchone())
    if p is None:
        raise HTTPException(404, "no such user photo")
    with db.tx(conn):
        conn.execute("UPDATE objects SET primary_photo_id=NULL WHERE id=? AND primary_photo_id=?", (oid, pid))
        conn.execute("DELETE FROM photos WHERE id=?", (pid,))
        db.log_event(conn, oid, "user", "photo_removed", {"id": pid, "path": p["path"]}, None)
    for rel in (p["path"], photos.thumb_of(p["path"])):
        try:
            os.remove(_data_file(rel))
        except (HTTPException, OSError):
            pass
    return {"ok": True}


@router.post("/api/objects/{oid}/approve")
def approve(oid: str, body: dict = Body(...)):
    """what=identity: draft -> identified. what=model: pick a candidate/asset -> verified."""
    conn = _conn()
    o = db.get_object(conn, oid)
    if o is None:
        raise HTTPException(404, "no such object")
    what = body.get("what")
    if what == "identity":
        new = "identified" if o["status"] == "draft" else o["status"]
        db.update_object(conn, oid, {"status": new, "confidence": 1.0}, actor="user", reason="identity approved")
    elif what == "model":
        cid = body.get("candidate_id")
        has_asset = conn.execute("SELECT 1 FROM assets WHERE object_id=? AND kind IN ('usd','glb','obj')",
                                 (oid,)).fetchone()
        if cid is not None:
            if not conn.execute("SELECT 1 FROM candidates WHERE id=? AND object_id=?", (cid, oid)).fetchone():
                raise HTTPException(404, "no such candidate")
            with db.tx(conn):
                conn.execute("UPDATE candidates SET chosen=(id=?) WHERE object_id=?", (cid, oid))
                db.log_event(conn, oid, "user", "candidate_chosen", None, {"candidate_id": cid})
        elif not has_asset:
            raise HTTPException(400, "nothing to approve: no candidate_id and no normalized asset")
        db.update_object(conn, oid, {"status": "verified"}, actor="user", reason="model approved")
    else:
        raise HTTPException(400, "what must be 'identity' or 'model'")
    return _full(conn, oid)


@router.post("/api/objects/{oid}/check")
def check(oid: str, body: dict = Body(...)):
    """Collection check: have=true -> owned (seed becomes given, searchable); have=false -> not owned."""
    conn = _conn()
    o = db.get_object(conn, oid)
    if o is None:
        raise HTTPException(404, "no such object")
    if not isinstance(body.get("have"), bool):
        raise HTTPException(400, "have must be true or false")
    ch = {"owned": body["have"], "collection_checked_at": db.now()}
    if body["have"]:
        ch["include_in_search"] = True
        if o["origin"] == "seed":
            ch["origin"] = "given"
    else:
        ch["include_in_search"] = False
    db.update_object(conn, oid, ch, actor="user", reason="collection check")
    return _full(conn, oid)


@router.post("/api/objects/{oid}/refimages")
def fetch_refimages(oid: str, body: dict = Body(default={})):
    """Search Openverse for more reference images (optionally with custom text). Takes a few seconds."""
    from objsearch import refimages
    conn = _conn()
    if db.get_object(conn, oid) is None:
        raise HTTPException(404, "no such object")
    q = (body.get("query") or "").strip() or None
    if q and len(q) > 120:
        raise HTTPException(400, "query too long")
    try:
        with track.run(conn, "refimages", provider="openverse", notes=f"dashboard: {oid}"):
            got = refimages.fetch_for_object(conn, oid, n=min(int(body.get("n", 3)), 6), query=q, force=True,
                                             pace=False)
    except track.QuotaExceeded as e:
        raise HTTPException(429, str(e))
    except Exception as e:
        raise HTTPException(502, f"image search failed: {type(e).__name__}: {e}")
    o = _full(conn, oid)
    o["fetched"] = got
    return o


@router.post("/api/objects/{oid}/refimages/{rid}/choose")
def choose_refimage(oid: str, rid: int):
    conn = _conn()
    if not conn.execute("SELECT 1 FROM ref_images WHERE id=? AND object_id=?", (rid, oid)).fetchone():
        raise HTTPException(404, "no such reference image")
    db.choose_ref_image(conn, oid, rid)
    return _full(conn, oid)


@router.delete("/api/objects/{oid}/refimages/{rid}")
def delete_refimage(oid: str, rid: int):
    """Drop a wrong reference image (file + row); the next best one becomes chosen."""
    conn = _conn()
    r = db.delete_ref_image(conn, oid, rid)
    if r is None:
        raise HTTPException(404, "no such reference image")
    for rel in (r["path"], photos.thumb_of(r["path"])):
        try:
            os.remove(_data_file(rel))
        except (HTTPException, OSError):
            pass
    return _full(conn, oid)


@router.post("/api/objects/{oid}/draft")
def draft(oid: str):
    raise HTTPException(501, "The research agent arrives in phase 4 (Gemini free tier). "
                             "Try the fake pipeline on the Runs page meanwhile.")


# ---------------------------------------------------------------- runs / tracking API

@router.get("/api/runs")
def list_runs(limit: int = Query(100, le=1000)):
    return {"runs": db.list_runs(_conn(), limit)}


@router.get("/api/runs/{run_id}")
def get_run(run_id: str):
    r = db.get_run(_conn(), run_id)
    if r is None:
        raise HTTPException(404, "no such run")
    return r


@router.get("/api/runs/{run_id}/log")
def run_log(run_id: str, lines: int = Query(400, le=5000)):
    r = db.get_run(_conn(), run_id)
    if r is None or not r.get("log_path"):
        raise HTTPException(404, "no log")
    try:
        with open(_data_file(r["log_path"]), encoding="utf-8") as f:
            tail = f.readlines()[-lines:]
    except HTTPException:
        tail = []
    return Response("".join(tail), media_type="text/plain")


@router.get("/api/text")
def text_file(path: str, lines: int = Query(2000, le=20000)):
    """Transcripts/tracebacks inside runs/ as plain text."""
    if not path.startswith("runs/") or os.path.splitext(path)[1] not in (".jsonl", ".txt", ".json"):
        raise HTTPException(400, "only run text files")
    with open(_data_file(path), encoding="utf-8") as f:
        return Response("".join(f.readlines()[:lines]), media_type="text/plain")


_demo_lock = threading.Lock()


@router.post("/api/runs/demo", status_code=202)
def start_demo(body: dict = Body(default={})):
    fail = body.get("fail_at")
    if fail not in (None, "identify", "acquire", "convert"):
        raise HTTPException(400, "fail_at must be identify|acquire|convert")
    if not _demo_lock.acquire(blocking=False):
        raise HTTPException(409, "a demo run is already in progress")

    def work():
        from objsearch import demo
        conn = db.connect()
        try:
            demo.run_demo(conn, fail)
        finally:
            conn.close()
            _demo_lock.release()

    threading.Thread(target=work, daemon=True).start()
    return {"started": True, "fail_at": fail}


@router.get("/api/activity")
def activity(limit: int = Query(50, le=500)):
    return db.activity(_conn(), limit)


@router.get("/api/quota")
def quota():
    return track.quota_status(_conn())


@router.get("/api/usage")
def usage():
    return db.usage_totals(_conn())


# ---------------------------------------------------------------- identify from photos

_identify_lock = threading.Lock()


@router.get("/api/identify/status")
def identify_status():
    from objsearch import identify
    st = identify.backend_status()
    st["busy"] = _identify_lock.locked()
    return st


@router.post("/api/identify/check")
def identify_check():
    """One tiny Gemini request: is the key usable (billing, model access)?"""
    from objsearch.llm import gemini_backend
    return gemini_backend.check_key()


@router.get("/api/identify")
def identify_recent(limit: int = Query(20, le=100)):
    from objsearch import identify
    return {"runs": identify.recent(_conn(), limit)}


@router.post("/api/identify", status_code=202)
async def identify_start(files: list[UploadFile] = File(...), hint: str = Form(""), dims: str = Form("")):
    """Start identifying ONE object from 1-6 photos; poll GET /api/identify/{run_id}."""
    from objsearch import identify
    if not 1 <= len(files) <= 6:
        raise HTTPException(400, "send 1 to 6 photos of the same object")
    images = []
    for f in files:
        data = await f.read(photos.MAX_UPLOAD + 1)
        if len(data) > photos.MAX_UPLOAD:
            raise HTTPException(400, f"{f.filename}: larger than 25 MB")
        try:
            images.append(identify._load(data))
        except ValueError as e:
            raise HTTPException(400, f"{f.filename}: {e}")
    dims_mm = None
    if dims.strip():
        try:
            dims_mm = [float(x) for x in dims.replace(",", " ").split()]
            assert len(dims_mm) == 3 and all(x > 0 for x in dims_mm)
        except (ValueError, AssertionError):
            raise HTTPException(400, "dims must be three positive numbers (mm)")
    if not _identify_lock.acquire(blocking=False):
        raise HTTPException(409, "an identification is already running; wait for it to finish")
    started, box = threading.Event(), {}

    def work():
        conn = db.connect()
        try:
            res = identify.identify(conn, images, hint=hint.strip()[:300] or None, dims_mm=dims_mm,
                                    on_start=lambda rid: (box.update(run_id=rid), started.set()))
            from objsearch import acquire
            if res.get("object_id") and acquire.auto_enabled():  # build the 3D model now, before you even confirm
                acquire.acquire(conn, res["object_id"])
                from objsearch import docs
                if docs.auto_enabled():  # then datasheet + mass/materials -> physics in sim.usda
                    docs.safe_research(conn, res["object_id"])
        except Exception as e:  # run + failed step are recorded; surface errors that happen before it exists
            box["error"] = f"{type(e).__name__}: {e}"
        finally:
            started.set()
            conn.close()
            _identify_lock.release()

    threading.Thread(target=work, daemon=True).start()
    started.wait(15)
    if "run_id" not in box:
        raise HTTPException(500, box.get("error", "identification did not start"))
    return {"run_id": box["run_id"]}


@router.get("/api/identify/{run_id}")
def identify_get(run_id: str):
    from objsearch import identify
    v = identify.run_view(_conn(), run_id)
    if v is None:
        raise HTTPException(404, "no such identify run")
    return v


@router.post("/api/identify/{run_id}/confirm")
def identify_confirm(run_id: str, body: dict = Body(...)):
    from objsearch import identify
    if not isinstance(body.get("correct"), bool):
        raise HTTPException(400, "correct must be true or false")
    name = (body.get("name") or "").strip() or None
    if name and len(name) > 120:
        raise HTTPException(400, "name too long")
    try:
        out = identify.confirm(_conn(), run_id, body["correct"], owned=bool(body.get("owned", True)), name=name)
    except KeyError:
        raise HTTPException(404, "no such identify run (or it has no result yet)")
    except ValueError as e:
        raise HTTPException(409, str(e))
    return {**out, "view": identify.run_view(_conn(), run_id)}


# ---------------------------------------------------------------- 3D models

_model_lock = threading.Lock()


@router.post("/api/objects/{oid}/model", status_code=202)
async def add_model(oid: str, file: UploadFile | None = File(None), url: str = Form(""), license: str = Form(""),
                    source_url: str = Form("")):
    """Attach a 3D model from an uploaded file or a URL; runs convert + Blender in the background."""
    from objsearch import models
    from objsearch.blender import interface as blender
    conn = _conn()
    if db.get_object(conn, oid) is None:
        raise HTTPException(404, "no such object")
    if not blender.available():
        raise HTTPException(503, "Blender (.venv-bpy) is not installed")
    if bool(file and file.filename) == bool(url.strip()):
        raise HTTPException(400, "send either a file or a url")
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in models.MESH_EXT:
            raise HTTPException(400, f"unsupported model type {ext or '?'}; allowed: {sorted(models.MESH_EXT)}")
        data = await file.read(models.MAX_BYTES + 1)
        if len(data) > models.MAX_BYTES:
            raise HTTPException(400, "model larger than 150 MB")
        up = os.path.join(config.data_dir(), "quarantine", oid, "upload" + ext)
        os.makedirs(os.path.dirname(up), exist_ok=True)
        with open(up, "wb") as f:
            f.write(data)
        src = up
    else:
        src = url.strip()
        if not src.startswith(("http://", "https://")):
            raise HTTPException(400, "url must be http(s)")
    if not _model_lock.acquire(blocking=False):
        raise HTTPException(409, "a model is already being processed")
    started, box = threading.Event(), {}

    def work():
        c = db.connect()
        try:
            models.add_model(c, oid, src, license=license.strip() or None, source_url=source_url.strip() or None,
                             on_start=lambda rid: (box.update(run_id=rid), started.set()))
        except Exception as e:
            box["error"] = f"{type(e).__name__}: {e}"
        finally:
            started.set()
            c.close()
            _model_lock.release()

    threading.Thread(target=work, daemon=True).start()
    started.wait(30)
    if "run_id" not in box:
        raise HTTPException(500, box.get("error", "model processing did not start"))
    return {"run_id": box["run_id"]}


@router.delete("/api/objects/{oid}/model")
def remove_model(oid: str):
    from objsearch import models
    conn = _conn()
    if db.get_object(conn, oid) is None:
        raise HTTPException(404, "no such object")
    n = models.remove_models(conn, oid, reason="removed in dashboard")
    return {"removed_assets": n, "object": _full(conn, oid)}


# ---------------------------------------------------------------- 3D parts library

_parts_lock = threading.Lock()


@router.get("/api/parts")
def parts_list():
    from objsearch import kicad, parts
    conn = _conn()
    rows = parts.listing(conn)
    names = {r[0]: r[1] for r in conn.execute("SELECT id, name FROM objects")}
    for r in rows:
        r["used_by"] = [{"id": i, "name": names.get(i, i)} for i in r.get("seen_in") or []]
        r["footprint"] = parts.footprint_for(conn, r["key"])
    counts = {s: sum(r["status"] == s for r in rows) for s in ("real", "approximate", "missing", "unmapped")}
    return {"parts": rows, "counts": counts, "index_ready": kicad.index_ready(), "busy": _parts_lock.locked()}


def _parts_job(fn):
    """Run fn(conn, on_start) in the background; return its run id once it exists."""
    if not _parts_lock.acquire(blocking=False):
        raise HTTPException(409, "a parts job is already running")
    started, box = threading.Event(), {}

    def work():
        c = db.connect()
        try:
            with track.run(c, "parts", notes="parts library job") as r:
                box["run_id"] = r.id
                started.set()
                box["result"] = fn(c)
        except Exception as e:
            box["error"] = f"{type(e).__name__}: {e}"
        finally:
            started.set()
            c.close()
            _parts_lock.release()

    threading.Thread(target=work, daemon=True).start()
    started.wait(30)
    if "run_id" not in box:
        raise HTTPException(500, box.get("error", "job did not start"))
    return {"run_id": box["run_id"]}


@router.post("/api/parts/real", status_code=202)
async def parts_add_real(key: str = Form(...), file: UploadFile | None = File(None), url: str = Form(""),
                         license: str = Form(""), source_url: str = Form(""), rebuild: bool = Form(True)):
    """Give a library part a real STEP (file or link); auto-aligned to its footprint; optionally rebuild users."""
    from objsearch import models, parts
    conn = _conn()
    if parts.get(conn, key) is None:
        raise HTTPException(404, "no such library part")
    if not license.strip():
        raise HTTPException(400, "license is required (as stated where you got the file)")
    if bool(file and file.filename) == bool(url.strip()):
        raise HTTPException(400, "send either a STEP file or a url")
    if file and file.filename:
        if os.path.splitext(file.filename)[1].lower() not in (".step", ".stp"):
            raise HTTPException(400, "parts must be STEP files (.step/.stp)")
        data = await file.read(models.MAX_BYTES + 1)
        if not data.lstrip()[:20].startswith(b"ISO-10303-21"):
            raise HTTPException(400, "not a STEP file")
        src = os.path.join(config.data_dir(), "quarantine", "_parts", "upload.step")
        os.makedirs(os.path.dirname(src), exist_ok=True)
        with open(src, "wb") as f:
            f.write(data)
    else:
        src = url.strip()
        if not src.startswith(("http://", "https://")):
            raise HTTPException(400, "url must be http(s)")

    def job(c):
        path = src
        if path.startswith(("http://", "https://")):
            path, _, _ = models.fetch("_parts", path)
        p = parts.add_real(c, key, path, license.strip(), source_url.strip() or (url.strip() or None))
        rebuilt = []
        if rebuild:
            for oid in p.get("seen_in") or []:
                try:
                    models.rebuild(c, oid)
                    rebuilt.append(oid)
                except (ValueError, KeyError) as e:
                    track.log.warning(f"rebuild {oid} skipped: {e}")
        return {"part": p["key"], "rebuilt": rebuilt}
    return _parts_job(job)


@router.post("/api/parts/rebuild", status_code=202)
def parts_rebuild(body: dict = Body(default={})):
    from objsearch import models, parts
    oid = body.get("object_id")

    def job(c):
        done = []
        for o in [oid] if oid else parts.affected_objects(c):
            models.rebuild(c, o)
            done.append(o)
        return done
    return _parts_job(job)


@router.get("/api/parts/candidates")
def parts_candidates(key: str):
    """Real-STEP candidates for a part from open CAD libraries (FreeCAD Parts Library), best first."""
    from objsearch import parts
    if parts.get(_conn(), key) is None:
        raise HTTPException(404, "no such library part")
    try:
        return {"candidates": parts.search_open_libraries(key)}
    except Exception as e:
        raise HTTPException(502, f"library search failed: {type(e).__name__}: {e}")
