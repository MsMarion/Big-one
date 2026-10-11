"""Fake end-to-end run: same stages, tracking and DB writes as the real pipeline, but with placeholder
observations, a scripted FakeLLM and canned tools. Populates the Runs/Objects dashboard without any
API key, camera or Blender. Demo rows use ids starting with 'demo-' and are removed by clean().

    python -m objsearch demo [--fail-at acquire|convert]
"""
import logging
import os
import shutil

from . import config, db, fakes, track
from .llm import tools
from .llm.base import Budget, Image
from .llm.fake_backend import FakeLLM

log = logging.getLogger("objsearch.demo")

INSTANCES = [  # what observe would produce from a frame: dims (mm), colors, OCR text
    {"label": "blue module, 8-pin header", "dims": [24.8, 14.3, 11.0], "color": (40, 70, 160), "ocr": "ESP-01"},
    {"label": "red brick, 2x4 studs", "dims": [31.8, 15.8, 11.4], "color": (200, 30, 30), "ocr": ""},
    {"label": "black camera with clip", "dims": [72.0, 31.0, 30.0], "color": (25, 25, 28), "ocr": "HD 1080P"},
]


def _match(conn, dims, tol=0.08):
    """Dims-only stand-in for retrieval (the real one adds DINO embeddings)."""
    want, best = sorted(dims), (None, 0.0)
    for o in db.rows(conn.execute("SELECT id, dims_mm FROM objects WHERE dims_mm IS NOT NULL "
                                  "AND status!='rejected' AND id NOT LIKE 'demo-%'")):
        have = sorted(o["dims_mm"])
        if len(have) != 3:
            continue
        err = max(abs(a - b) / b for a, b in zip(want, have))
        if err <= tol and 1 - err > best[1]:
            best = (o["id"], round(1 - err, 3))
    return best


def identify_script(fail):
    s = [{"tool_calls": [{"name": "catalog_lookup", "args": {"text": "usb webcam"}}],
          "text": "Black clip-on camera, 72 mm wide, 'HD 1080P' print. Check the catalog first."},
         {"tool_calls": [{"name": "web_search", "args": {"query": "black usb webcam HD 1080P clip 72 mm"}}]},
         {"tool_calls": [{"name": "web_fetch", "args": {"url": "https://example.test/webcam-1080p"}}]},
         {"final": {"name": "Generic 1080p USB webcam (demo)", "category": "camera", "manufacturer": "unknown",
                    "mpn": "", "aliases": ["usb webcam", "1080p webcam"], "expected_dims_mm": [72, 31, 30],
                    "evidence_urls": ["https://example.test/webcam-1080p"], "confidence": 0.74, "catalog_id": "",
                    "reasoning": "Size, clip and 'HD 1080P' print match the product page."}}]
    if fail == "identify":
        s.insert(2, {"raise": "ConnectionError: 503 from provider (fake)"})
    return s


def acquire_script(fail):
    s = [{"tool_calls": [{"name": "sketchfab_search", "args": {"query": "usb webcam"}}]},
         {"tool_calls": [{"name": "sketchfab_download", "args": {"uid": "fakeuid01"}}]},
         {"tool_calls": [{"name": "convert", "args": {"file_id": "file1"}}]},
         {"tool_calls": [{"name": "compare", "args": {"model_id": "model1"}}]},
         {"final": {"candidates": [{"model_id": "model1", "source_url": "https://example.test/models/fakeuid01",
                                    "license": "CC-BY-4.0", "score": 0.81}],
                    "notes": "Only CC model with matching proportions."}}]
    if fail == "acquire":
        s.insert(2, {"raise": "ConnectionError: 503 from provider mid-acquire (fake)"})
    if fail == "convert":  # tool fails inside its nested step; the agent sees the error and gives up
        s[3:] = [{"final": {"candidates": [], "notes": "Conversion failed; no usable model."}}]
    return s


def run_demo(conn=None, fail_at=None):
    conn = conn or db.connect()
    cfg = config.load()
    with track.run(conn, "demo", provider="fake", model="fake", frame_dir="demo://synthetic",
                   notes=f"fake demo run{f' (fail_at={fail_at})' if fail_at else ''}") as r:
        with track.step("observe", inputs={"frame": "demo://synthetic"}) as s:
            sightings = []
            for i, inst in enumerate(INSTANCES):
                crop = fakes.make_crop(r.path("crops", f"inst_{i}.jpg"), inst["label"], inst["color"])
                sid = db.add_sighting(conn, r.id, i, crop_path=track.rel(crop), observed_dims_mm=inst["dims"],
                                      ocr_text=inst["ocr"], status="unknown")
                sightings.append((sid, inst, crop))
            s.set(instances=len(sightings))

        for sid, inst, crop in sightings:
            try:
                with track.step("retrieve", sighting_id=sid, inputs={"dims_mm": inst["dims"]}) as s:
                    oid, score = _match(conn, inst["dims"])
                    s.set(match=oid, score=score)
                if oid:
                    conn.execute("UPDATE sightings SET object_id=?, match_method='retrieval', score=?, "
                                 "status='matched' WHERE id=?", (oid, score, sid))
                    continue
                _research(conn, cfg, r, sid, inst, crop, fail_at)
            except Exception as e:  # one bad object must not end the run
                log.warning(f"instance {inst['label']!r} failed: {type(e).__name__}: {e}")

        with track.step("assemble") as s:
            s.skip("scene.usd needs Blender (phase 2)")
    return r.id


def _research(conn, cfg, r, sid, inst, crop, fail_at):
    handlers = fakes.fake_handlers(conn, raise_in_step="convert" if fail_at == "convert" else None)
    content = [f"Observed object: dims_mm={inst['dims']}, OCR={inst['ocr']!r}, description={inst['label']!r}.",
               Image(path=crop, label="crop_0")]
    ident = None
    with track.step("identify", sighting_id=sid) as s:
        res = FakeLLM(identify_script(fail_at)).run(
            "Identify the object.", content, tools.build_tools(handlers, tools.IDENTIFY_TOOLS),
            tools.IDENTIFICATION_SCHEMA, Budget.from_config(cfg))
        s.set(stop_reason=res.stop_reason, budget=res.budget)
        if not res.output:
            s.skip(f"no identification ({res.stop_reason})")
        ident = res.output
        oid = "demo-" + db.slugify(ident["name"])
        if not db.get_object(conn, oid):
            db.create_object(conn, {"name": ident["name"], "category": ident["category"],
                                    "manufacturer": ident.get("manufacturer"), "mpn": ident.get("mpn") or None,
                                    "aliases": ident.get("aliases"), "dims_mm": ident.get("expected_dims_mm"),
                                    "origin": "found", "status": "identified", "confidence": ident["confidence"],
                                    "notes": "Created by the fake demo run.", "owned": False,
                                    "include_in_search": False},
                             actor="agent", reason="identified by demo agent", run_id=r.id, oid=oid)
            for u in ident.get("evidence_urls", []):
                db.add_evidence(conn, oid, u, kind="product", trust="fake")
            db.add_photo(conn, oid, track.rel(crop), kind="crop", source="run", actor="agent")
        conn.execute("UPDATE sightings SET object_id=?, match_method='agent', score=?, status='matched' WHERE id=?",
                     (oid, ident["confidence"], sid))
        s.object_id = oid
        s.set(object_id=oid)
    if ident is None:  # identify was skipped
        return

    with track.step("acquire", object_id=oid, sighting_id=sid) as s:
        res = FakeLLM(acquire_script(fail_at)).run(
            "Find a realistic 3D model.", content, tools.build_tools(handlers, tools.ACQUIRE_TOOLS),
            tools.ACQUIRE_SCHEMA, Budget.from_config(cfg))
        s.set(stop_reason=res.stop_reason, budget=res.budget)
        for rank, c in enumerate((res.output or {}).get("candidates", []), 1):
            db.add_candidate(conn, oid, sighting_id=sid, source_url=c["source_url"], license=c["license"],
                             scores={"score": c.get("score")}, rank=rank)
        if res.output and res.output["candidates"]:
            db.update_object(conn, oid, {"status": "model_found"}, actor="agent", reason="demo candidate found",
                             run_id=r.id)

    with track.step("normalize", object_id=oid) as s:
        s.skip("needs bpy (phase 2)")


def clean(conn=None):
    """Remove demo runs, their files, and demo-* objects."""
    conn = conn or db.connect()
    ids = [r[0] for r in conn.execute("SELECT id FROM runs WHERE kind='demo'")]
    with db.tx(conn):
        for rid in ids:
            conn.execute("DELETE FROM runs WHERE id=?", (rid,))  # cascades steps, calls, sightings
        conn.execute("DELETE FROM object_events WHERE object_id LIKE 'demo-%'")
        conn.execute("DELETE FROM objects WHERE id LIKE 'demo-%'")
    for rid in ids:
        shutil.rmtree(os.path.join(config.data_dir(), "runs", rid), ignore_errors=True)
    return len(ids)
