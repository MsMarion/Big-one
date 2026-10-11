"""Identify an object from plain photos (no depth needed): upload 1-6 photos of ONE object, get back a
catalog match or a newly researched object, then confirm or correct it.

    python -m objsearch identify photo.jpg [more angles...] [--hint "blue board, says ESP-01"] [--dims 25 14 11]
    python -m objsearch identify folder/            # one folder = one object (several angles)

Stages (all tracked): intake -> identify (agent: catalog, Brickognize, Google-grounded search, page
fetch) -> record. Without a GEMINI_API_KEY it still works for LEGO via Brickognize (keyless mode).
"""
import io
import json
import logging
import os
import time

from PIL import Image, ImageOps

try:  # phone photos (iPhone HEIC)
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from . import config, db, photos, track
from .llm import get_backend, tools
from .llm.base import Budget
from .llm.base import Image as LLMImage
from .sources import brickognize, ldraw, web

log = logging.getLogger("objsearch.identify")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}

SYSTEM = """You identify ONE physical object from photos for a maker/robotics parts catalog \
(microcontroller boards and modules, sensors, cameras, storage, LEGO, small electronics).

How to work:
1. Look closely first: shape, colors, connectors, pin headers, antennas, and read ALL printed text \
(chip markings, silkscreen, labels, part numbers, FCC IDs). Use any size the user gives.
2. If it is one of the catalog items listed below, use that item's id as catalog_id (match the exact \
variant: e.g. ESP-01 vs ESP-12, Uno vs Mega).
3. Otherwise research it: web_search with specific queries (markings, part numbers, "<name> dimensions"), \
web_fetch pages that look authoritative. For LEGO pieces call brickognize_identify on the photo.
4. Call submit_result exactly once. Use the specific product name. Many maker boards are clones: do not \
call a board "official" or "genuine" unless the photo shows it (a blue Uno marked "Made in Italy" is usually \
a clone); name the design, e.g. "Arduino Uno R3 (or compatible clone)", and say what suggests a clone. \
Put datasheet dimensions [L, W, H] in mm \
in expected_dims_mm only if a source states them. Only list evidence_urls you actually saw. Calibrate \
confidence honestly; a low-confidence answer is better than invented details."""


def _load(img):
    """bytes or path -> normalized JPEG bytes (EXIF-rotated, <= 1600 px, metadata stripped)."""
    data = img if isinstance(img, (bytes, bytearray)) else open(img, "rb").read()
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception as e:
        raise ValueError(f"not a readable image: {e}") from None
    im.thumbnail((1600, 1600))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def expand_paths(paths):
    """CLI helper: files stay files; a folder becomes its image files (one object, several angles)."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, f) for f in os.listdir(p) if os.path.splitext(f)[1].lower() in IMAGE_EXT)
        else:
            out.append(p)
    return out


def catalog_brief(conn):
    rows = db.list_objects(conn)
    lines = []
    for o in rows:
        if o["id"].startswith("demo-"):
            continue
        dims = "x".join(f"{d:g}" for d in o["dims_mm"]) + "mm" if o.get("dims_mm") else "?"
        al = ", ".join((o.get("aliases") or [])[:3])
        lines.append(f"{o['id']} | {o['name']} | {dims}" + (f" | {al}" if al else "") +
                     (f" | LEGO {o['ldraw_id']}" if o.get("ldraw_id") else ""))
    return "\n".join(lines)


def _collect_urls(obj, seen):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("url", "uri") and isinstance(v, str):
                seen.add(v)
            else:
                _collect_urls(v, seen)
    elif isinstance(obj, list):
        for v in obj:
            _collect_urls(v, seen)


def _handlers(conn, backend, images, seen_urls):
    """Tool handlers for the identify agent. Every URL a tool returns (or fetches) goes into seen_urls,
    so evidence can be limited to pages the agent really saw."""
    by_label = {f"photo_{i + 1}": b for i, b in enumerate(images)}

    def seen(fn):
        def wrapped(a):
            out = fn(a)
            _collect_urls(out, seen_urls)
            return out
        return wrapped

    def brick(a):
        data = by_label.get(a["image"]) or images[0]
        return brickognize.identify(data)

    def ldraw_get(a):
        info = ldraw.find_part(a["part"])
        return info or {"error": f"part {a['part']} not found in the official LDraw library"}

    h = {"catalog_lookup": tools.catalog_lookup_handler(conn), "brickognize_identify": seen(brick),
         "ldraw_get": seen(ldraw_get), "web_fetch": seen(lambda a: web.fetch_text(a["url"]))}
    if hasattr(backend, "grounded_search"):
        h["web_search"] = seen(lambda a: backend.grounded_search(a["query"]))
    return h


def backend_status():
    """What identify will use right now (for the dashboard)."""
    from .llm.gemini_backend import api_key
    cfg = config.load()
    prov = cfg.get("provider", "gemini")
    has_key = bool(api_key()) if prov == "gemini" else prov == "fake"
    return {"provider": prov, "model": cfg["models"].get(prov, {}).get("agent") if prov in cfg["models"] else prov,
            "search_model": cfg["models"].get("gemini", {}).get("search"), "ready": has_key,
            "mode": "agent" if has_key else "keyless (LEGO via Brickognize only)"}


def identify(conn, images, hint=None, dims_mm=None, provider=None, backend=None, on_start=None):
    """images: list of bytes or paths of ONE object. Returns the result dict (also stored on the run).
    on_start(run_id) is called as soon as the run exists (the dashboard polls it while the agent works)."""
    images = [_load(i) for i in images][:6]
    if not images:
        raise ValueError("no images")
    cfg = config.load()
    status = backend_status()
    provider = provider or (backend.provider if backend else status["provider"])
    if backend is None and status["ready"] and provider == "gemini":
        backend = get_backend("gemini")
    with track.run(conn, "identify", provider=provider if backend else "keyless",
                   model=backend.model if backend else "brickognize", frame_dir="upload",
                   notes=(hint or "")[:200] or None) as r:
        if on_start:
            on_start(r.id)
        with track.step("intake", inputs={"images": len(images), "hint": hint, "dims_mm": dims_mm}) as s:
            paths = []
            for i, data in enumerate(images):
                p = r.path("inputs", f"photo_{i + 1}.jpg")
                with open(p, "wb") as f:
                    f.write(data)
                paths.append(track.rel(p))
            sid = db.add_sighting(conn, r.id, 0, crop_path=paths[0], observed_dims_mm=dims_mm, status="unknown")
            s.set(sighting_id=sid, photos=paths)

        ident, stop = None, None
        out = {}
        try:
            _identify_step(conn, cfg, sid, backend, images, hint, dims_mm, out)
            ident, stop = out.get("ident"), out.get("stop")
        except track.QuotaExceeded as e:
            stop = f"quota: {e}"
        except Exception as e:  # the failed step (with traceback) is in the run; still record an outcome
            stop = f"error: {type(e).__name__}: {e}"

        with track.step("record", sighting_id=sid) as s:
            result = _record(conn, r.id, sid, ident, paths, stop)
            s.set(**result)
    return result


def _identify_step(conn, cfg, sid, backend, images, hint, dims_mm, out):
    """Agent (or keyless Brickognize) identification; results go into `out` so the caller can still
    record an outcome if this step raises."""
    ident = stop = None
    with track.step("identify", sighting_id=sid) as s:
        if backend is not None:
            content = [f"Catalog (id | name | size | aliases):\n{catalog_brief(conn)}"]
            if hint:
                content.append(f"User note: {hint}")
            if dims_mm:
                content.append(f"Measured size (mm): {dims_mm}")
            content.append(f"{len(images)} photo(s) of the same object follow.")
            content += [LLMImage(data=b, label=f"photo_{i + 1}") for i, b in enumerate(images)]
            seen_urls = set()
            res = backend.run(SYSTEM, content, tools.build_tools(_handlers(conn, backend, images, seen_urls),
                                                                 tools.IDENTIFY_TOOLS),
                              tools.IDENTIFICATION_SCHEMA, Budget.from_config(cfg))
            ident, stop = res.output, res.stop_reason
            if ident:  # evidence = only pages a tool actually returned/fetched this run
                urls = [u for u in ident.get("evidence_urls") or [] if isinstance(u, str)]
                ident["evidence_urls"] = [u for u in urls if u in seen_urls]
                ident["unverified_urls"] = [u for u in urls if u not in seen_urls]
            s.set(mode="agent", stop_reason=stop, budget=res.budget, errors=res.errors,
                  unverified_urls=(ident or {}).get("unverified_urls", []))
        else:
            s.set(mode="keyless")
            t0 = time.perf_counter()
            br = brickognize.identify(images[0])
            track.record_tool_call("brickognize_identify", {"image": "photo_1"}, br["items"][:3], False,
                                   round((time.perf_counter() - t0) * 1e3))
            top = br["items"][0] if br["items"] else None
            s.set(brickognize=br["items"][:3])
            if top and top["score"] >= 0.5 and top["type"] == "part":
                ident = {"name": f"LEGO {top['name']}", "category": "lego", "manufacturer": "LEGO",
                         "ldraw_id": top["id"], "catalog_id": "", "confidence": min(0.95, top["score"]),
                         "evidence_urls": [top["url"]] if top.get("url") else [], "aliases": [],
                         "reasoning": f"Brickognize match {top['id']} (score {top['score']}). "
                                      f"No GEMINI_API_KEY, so only LEGO can be identified."}
                stop = "keyless_brickognize"
            else:
                stop = "keyless_no_match"
                s.skip("no GEMINI_API_KEY set and Brickognize found no confident LEGO match")
    out.update(ident=ident, stop=stop)


def _record(conn, run_id, sid, ident, paths, stop):
    base = {"run_id": run_id, "sighting_id": sid, "photos": paths, "stop_reason": stop}
    if not ident:
        return {**base, "match": "none", "object_id": None, "name": None, "confidence": 0,
                "reasoning": "Could not identify the object."}
    conf = float(ident.get("confidence") or 0)
    oid, match = None, "new"
    cid = (ident.get("catalog_id") or "").strip()
    o = db.get_object(conn, cid) if cid else None
    if o and o["status"] != "rejected":
        oid, match = cid, "catalog"
    elif ident.get("ldraw_id"):
        hit = conn.execute("SELECT id FROM objects WHERE ldraw_id=? AND status!='rejected'",
                           (ident["ldraw_id"],)).fetchone()
        if hit:
            oid, match = hit[0], "catalog"
    if oid is None:
        same = conn.execute("SELECT id FROM objects WHERE id=? AND status!='rejected'",
                            (db.slugify(ident["name"]),)).fetchone()
        if same:
            oid, match = same[0], "catalog"
    if oid is None:
        dims = ident.get("expected_dims_mm")
        oid = db.create_object(conn, {
            "name": ident["name"][:120], "category": ident.get("category"),
            "manufacturer": ident.get("manufacturer") or None, "mpn": ident.get("mpn") or None,
            "ldraw_id": ident.get("ldraw_id") or None, "aliases": ident.get("aliases") or None,
            "dims_mm": dims if isinstance(dims, list) and len(dims) == 3 and all(d > 0 for d in dims) else None,
            "origin": "found", "status": "identified" if conf >= 0.6 else "draft", "confidence": conf,
            "notes": (ident.get("reasoning") or "")[:1000], "owned": False, "include_in_search": False},
            actor="agent", reason="identified from uploaded photo", run_id=run_id)
        for u in ident.get("evidence_urls") or []:
            if isinstance(u, str) and u.startswith(("http://", "https://")):
                db.add_evidence(conn, oid, u, kind="product", trust="seen by agent")
        for u in ident.get("unverified_urls") or []:
            if isinstance(u, str) and u.startswith(("http://", "https://")):
                db.add_evidence(conn, oid, u, kind="product", trust="unverified (from model memory)")
        db.add_photo(conn, oid, paths[0], kind="crop", source="run", actor="agent")
    conn.execute("UPDATE sightings SET object_id=?, match_method='agent', score=?, status='matched' WHERE id=?",
                 (oid, conf, sid))
    o = db.get_object(conn, oid)
    return {**base, "match": match, "object_id": oid, "name": o["name"], "confidence": conf,
            "reasoning": ident.get("reasoning"), "evidence_urls": ident.get("evidence_urls") or [],
            "unverified_urls": ident.get("unverified_urls") or [],
            "expected_dims_mm": ident.get("expected_dims_mm")}


def confirm(conn, run_id, correct, owned=True, name=None):
    """User feedback on an identify run. correct=True: keep (and add the photos as your photos if owned).
    correct=False: reject the guess; with name, file the photos under that (new or existing) object."""
    step = db.row(conn.execute("SELECT * FROM steps WHERE run_id=? AND stage='record' ORDER BY id DESC LIMIT 1",
                               (run_id,)).fetchone())
    if step is None:
        raise KeyError(run_id)
    res = step["outputs"]
    sid, oid, photo_paths = res["sighting_id"], res.get("object_id"), res.get("photos") or []
    st = conn.execute("SELECT status FROM sightings WHERE id=?", (sid,)).fetchone()
    if st and st[0] in ("confirmed", "rejected"):
        raise ValueError(f"this identification was already {st[0]}")
    if correct:
        if not oid:
            raise ValueError("nothing to confirm: the run found no object")
        conn.execute("UPDATE sightings SET status='confirmed' WHERE id=?", (sid,))
        o = db.get_object(conn, oid)
        ch = {"confidence": 1.0}
        if o["status"] == "draft":
            ch["status"] = "identified"
        if owned:
            ch.update(owned=True, include_in_search=True, collection_checked_at=db.now())
            if o["origin"] in ("seed", "found"):
                ch["origin"] = "given"
        db.update_object(conn, oid, ch, actor="user", reason="confirmed photo identification", run_id=run_id)
        if owned:
            _adopt_photos(conn, oid, photo_paths)
        return {"object_id": oid, "action": "confirmed"}
    conn.execute("UPDATE sightings SET status='rejected' WHERE id=?", (sid,))
    if oid and res.get("match") == "new":
        db.update_object(conn, oid, {"status": "rejected"}, actor="user", reason="wrong photo identification",
                         run_id=run_id)
    if not name:
        return {"object_id": None, "action": "rejected"}
    hit = db.search(conn, name, limit=1)
    target = hit[0]["id"] if hit and hit[0]["name"].lower() == name.strip().lower() else None
    if target is None:
        target = db.create_object(conn, {"name": name.strip()[:120], "origin": "given", "status": "identified",
                                         "owned": owned, "include_in_search": owned, "confidence": 1.0},
                                  actor="user", reason="named by user after photo identification", run_id=run_id)
    conn.execute("UPDATE sightings SET object_id=?, match_method='manual', status='confirmed' WHERE id=?",
                 (target, sid))
    if owned:
        _adopt_photos(conn, target, photo_paths)
    return {"object_id": target, "action": "corrected"}


def _adopt_photos(conn, oid, rel_paths):
    """Copy the run's photos in as the object's own (user) photos."""
    have = {p["path"] for p in db.rows(conn.execute("SELECT path FROM photos WHERE object_id=?", (oid,)))}
    for rel in rel_paths:
        full = os.path.join(config.data_dir(), rel)
        if os.path.exists(full):
            with open(full, "rb") as f:
                new = photos.ingest(oid, f.read())
            if new not in have:
                db.add_photo(conn, oid, new, kind="ref", source="user", actor="user")


def run_view(conn, run_id):
    """Everything the Identify page shows for one run: progress, agent activity, result."""
    r = db.get_run(conn, run_id)
    if r is None or r["kind"] != "identify":
        return None
    rec = next((st for st in reversed(r["steps"]) if st["stage"] == "record"), None)
    ident = next((st for st in r["steps"] if st["stage"] == "identify"), None)
    intake = next((st for st in r["steps"] if st["stage"] == "intake"), None)
    res = rec["outputs"] if rec and rec["status"] == "ok" else None
    view = {"run_id": run_id, "status": r["status"], "started_at": r["started_at"], "ended_at": r["ended_at"],
            "provider": r["provider"], "model": r["model"], "hint": r["notes"],
            "photos": (intake or {}).get("outputs", {}).get("photos", []),
            "steps": [{"stage": st["stage"], "status": st["status"], "ms": st["duration_ms"],
                       "error": st["error_msg"]} for st in r["steps"]],
            "activity": [{"tool": t["tool"], "args": t["args"], "is_error": bool(t["is_error"]),
                          "summary": (t["result_summary"] or "")[:300], "ms": t["latency_ms"]} for t in r["tool_calls"]],
            "llm_calls": len(r["llm_calls"]), "tokens": r["tokens_in"] + r["tokens_out"],
            "identify_notes": (ident or {}).get("outputs", {}), "result": res, "object": None, "sighting": None}
    if res and res.get("object_id"):
        o = db.get_object_full(conn, res["object_id"])
        if o:
            view["object"] = {k: o.get(k) for k in ("id", "name", "category", "dims_mm", "origin", "status", "owned",
                                                     "ref_path", "photo_path", "mpn", "ldraw_id")}
            view["model"] = model_status(conn, o, after=r["started_at"])
    if res:
        sg = conn.execute("SELECT status FROM sightings WHERE id=?", (res["sighting_id"],)).fetchone()
        view["sighting"] = sg[0] if sg else None
    return view


def recent(conn, limit=20):
    ids = [r[0] for r in conn.execute("SELECT id FROM runs WHERE kind='identify' ORDER BY started_at DESC LIMIT ?",
                                      (limit,))]
    out = []
    for rid in ids:
        v = run_view(conn, rid)
        out.append({k: v[k] for k in ("run_id", "status", "started_at", "photos", "result", "object", "sighting")})
    return out


def model_status(conn, o, after=None):
    """3D model state for the Identify page: building (acquire run in progress) / ready / none."""
    acq = db.row(conn.execute("SELECT id, status FROM runs WHERE kind='acquire' AND notes=? ORDER BY started_at DESC "
                              "LIMIT 1", (f"acquire {o['id']}",)).fetchone())
    chosen = next((c for c in o.get("candidates") or [] if c.get("chosen")), None)
    out = {"state": "none", "glb": o.get("glb_path"), "thumb": o.get("thumb_path"),
           "flags": ((chosen or {}).get("scores") or {}).get("flags") or [], "source": (chosen or {}).get("source_url"),
           "license": (chosen or {}).get("license"), "run_id": acq["id"] if acq else None, "how": None}
    if acq and acq["status"] == "running":
        out["state"] = "upgrading" if o.get("glb_path") else "building"  # placeholder shown while research runs
        step = conn.execute("SELECT stage FROM steps WHERE run_id=? ORDER BY id DESC LIMIT 1", (acq["id"],)).fetchone()
        out["step"] = step[0] if step else None
    elif o.get("glb_path"):
        out["state"] = "ready"
    if acq:
        how = conn.execute("SELECT outputs FROM steps WHERE run_id=? AND stage LIKE 'source_%' AND status='ok' "
                           "ORDER BY id DESC LIMIT 1", (acq["id"],)).fetchone()
        if how:
            out["how"] = json.loads(how[0] or "{}").get("how")
    return out
