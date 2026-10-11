"""Attach a 3D model to a catalog object: fetch (URL or local file) -> quarantine -> convert (STEP->GLB) ->
normalize in Blender (units, base on the table, centred) -> asset.usd / model.glb / mesh.obj / thumb.png.

    python -m objsearch model add arduino-uno-r3 https://.../Arduino%20Uno.stp --license "unknown" --source-url ...
    python -m objsearch model add lego-3001 C:/Downloads/brick.stl --license CC-BY-4.0

Downloads are untrusted data: allowlisted extensions only, size-capped, content-addressed by sha256,
never executed (.blend is refused: it can embed Python). License + source are stored with every asset.
"""
import hashlib
import os
import shutil
import time
import urllib.parse

import httpx

from . import config, db, track
from .blender import interface as blender
from .sources.openverse import UA

MESH_EXT = {".step", ".stp", ".glb", ".gltf", ".obj", ".stl", ".ply", ".fbx"}
BOARD_EXT = {".kicad_pcb", ".brd", ".pcbdoc", ".zip"}  # PCB designs, turned into 3D by objsearch.kicad
MAX_BYTES = 150 * 1024 * 1024


def _quarantine(oid):
    d = os.path.join(config.data_dir(), "quarantine", oid)
    os.makedirs(d, exist_ok=True)
    return d


def fetch(oid, src):
    """URL or local path -> (quarantined path, sha256, source_url)."""
    with track.step("download", object_id=oid, inputs={"src": src}) as s:
        is_url = urllib.parse.urlparse(src).scheme in ("http", "https")
        name = urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(src).path if is_url else src)) or "model"
        ext = os.path.splitext(name)[1].lower()
        if ext == ".blend":
            raise ValueError(".blend files are refused (they can contain Python scripts)")
        if ext not in MESH_EXT | BOARD_EXT:
            raise ValueError(f"unsupported model type {ext or '?'}; allowed: {sorted(MESH_EXT | BOARD_EXT)}")
        if is_url:
            t0 = time.perf_counter()
            with httpx.Client(timeout=120, headers={"User-Agent": UA}, follow_redirects=True) as c, \
                    c.stream("GET", src) as r:
                r.raise_for_status()
                chunks, n = [], 0
                for ch in r.iter_bytes():
                    n += len(ch)
                    if n > MAX_BYTES:
                        raise ValueError("model larger than 150 MB")
                    chunks.append(ch)
            data = b"".join(chunks)
            sha = hashlib.sha256(data).hexdigest()
            track.record_http("GET", src, r.status_code, len(data), sha, round((time.perf_counter() - t0) * 1e3))
        else:
            with open(src, "rb") as f:
                data = f.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ValueError("model larger than 150 MB")
            sha = hashlib.sha256(data).hexdigest()
        if ext in (".step", ".stp") and not data.lstrip()[:20].startswith(b"ISO-10303-21"):
            raise ValueError("file has a STEP extension but isn't a STEP file")
        dst = os.path.join(_quarantine(oid), f"{sha[:12]}{ext}")
        with open(dst, "wb") as f:
            f.write(data)
        s.set(path=track.rel(dst), sha256=sha, bytes=len(data))
    return dst, sha, src if is_url else None


def convert(oid, path, colors=None, extras=None):
    """PCB design -> STEP via KiCad (+ KiCad renders into extras["renders"]); STEP -> GLB with OpenCASCADE
    (cascadio); meshes pass through to Blender."""
    ext = os.path.splitext(path)[1].lower()
    if ext in BOARD_EXT:
        from . import kicad
        glb, info = kicad.board_to_step(path, os.path.splitext(path)[0] + "_kicad", colors=colors, object_id=oid,
                                        wait_renders=False, fmt="glb")  # KiCad writes GLB directly: no conversion
        if extras is not None:
            extras.update(info, srgb_colors=True)
            if extras.get("render_jobs"):
                extras["renders"] = kicad.finish_renders(extras.pop("render_jobs"))
        return glb
    if ext not in (".step", ".stp"):
        return path
    with track.step("convert", object_id=oid, inputs={"path": track.rel(path)}) as s:
        import cascadio
        out = os.path.splitext(path)[0] + ".glb"
        # 0.05 mm chordal tolerance: invisible at inventory scale, ~2x faster and lighter than 0.02
        rc = cascadio.step_to_glb(path, out, tol_linear=0.05, include_materials=True)
        if rc != 0 or not os.path.exists(out):
            raise RuntimeError(f"STEP conversion failed (cascadio rc={rc})")
        s.set(glb=track.rel(out), bytes=os.path.getsize(out))
    if extras is not None and extras.get("render_jobs"):
        from . import kicad
        extras["renders"] = kicad.finish_renders(extras.pop("render_jobs"))
    return out


def part_flags(rep):
    """KiCad build report -> human flags for parts that are approximate or missing in the 3D model."""
    flags = []
    for key in sorted(set(rep.get("approximate", []))):
        flags.append(f"approximate part: {key.split('/')[-1]} (box from its footprint outline; no real 3D model yet, "
                     f"add one on the Parts page)")
    for name, refs in sorted(rep.get("approximate_unmapped", {}).items()):
        flags.append(f"approximate part: {', '.join(refs)} ({name}) has no matching KiCad part; box from the board's "
                     f"outline (add a real STEP on the Parts page)")
    for name, libref in sorted(rep.get("suggested", {}).items()):
        flags.append(f"auto-matched part: {name} -> {libref} by pad geometry (check the render)")
    for name, refs in sorted(rep.get("unmapped", {}).items()):
        flags.append(f"missing part: {', '.join(refs)} ({name}) has no 3D model at all (no KiCad footprint mapped)")
    for m in sorted(set(rep.get("missing_model", []))):
        flags.append(f"missing part model: {m.split('/')[-1]}")
    return flags


PIPELINE_VERSION = "2026-10-11e"  # bump when build output changes, to invalidate the model cache


def build_key(conn, sha, colors, dims_mm, path, recolor=None):
    """Everything that changes a build's output -> cache key. Board builds also depend on the parts library and
    the footprint mapping, so improving the library automatically invalidates stale cached boards."""
    import hashlib
    import json
    parts = None
    if os.path.splitext(path)[1].lower() in BOARD_EXT:
        from . import kicad
        rows = conn.execute("SELECT key, status, path FROM parts3d ORDER BY key").fetchall()
        parts = [list(r) for r in rows] + [repr(sorted(kicad.EAGLE_TO_KICAD.items()))]
    blob = json.dumps([PIPELINE_VERSION, sha, colors, dims_mm, parts] + ([recolor] if recolor else []),
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _load_cached(out_dir):
    import json
    p = os.path.join(out_dir, "result.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        res = json.load(f)
    return res if all(os.path.exists(os.path.join(out_dir, v)) for v in res["files"].values()) else None


def _note_parts_use(conn, oid, parts_report):
    from . import parts as plib
    for key in parts_report.get("from_library", []) + parts_report.get("approximate", []):
        row = plib.get(conn, key)
        if row:
            plib.upsert(conn, key, row["kind"], row["status"], oid)
    for name in parts_report.get("approximate_unmapped", {}):
        row = plib.get(conn, f"eagle:{name}")
        if row:
            plib.upsert(conn, row["key"], row["kind"], row["status"], oid)


def _build(conn, o, oid, path, colors, out_dir, recolor=None):
    """convert -> normalize -> flags + KiCad renders, written into out_dir with result.json (the cache entry)."""
    import json
    shutil.rmtree(out_dir, ignore_errors=True)  # a half-written entry from an interrupted build
    extras = {}
    mesh = convert(oid, path, colors=colors, extras=extras)
    renders = extras.get("renders") or {}
    with track.step("normalize", object_id=oid, inputs={"src": track.rel(mesh), "dims_mm": o["dims_mm"]}) as s:
        from_step = os.path.splitext(path)[1].lower() in (".step", ".stp")
        res = blender.normalize(mesh, out_dir, o["dims_mm"], o["name"],
                                source_zup=from_step,      # cascadio GLB from STEP keeps CAD's raw Z-up
                                srgb_colors=from_step or extras.get("srgb_colors", False),  # CAD colors are sRGB
                                pad_finish=extras.get("copper_finish"),
                                recolor={k: v for k, v in (recolor or {}).items()} or None,
                                render=not renders)        # boards: KiCad already rendered (fast, accurate)
    if recolor:  # say exactly what was changed and on what evidence
        res["flags"] = (res.get("flags") or []) + [
            "recoloured parts (geometry unchanged): " + "; ".join(
                f"{v.get('label', k)} -> {v['hex']} ({v.get('why', 'per product photos')})" for k, v in recolor.items())]
        s.set(**{k: res[k] for k in ("dims_mm", "scale_factor", "tri_count", "flags") if k in res})
    # parts the build couldn't supply for real -> visible flags (and they're on the Parts library to-do list)
    rep = extras.get("models")
    if rep:
        res["flags"] = (res.get("flags") or []) + part_flags(rep)
        res["parts"] = {"placed": rep["placed"], "approximate": rep["approximate"],
                        "approximate_unmapped": rep.get("approximate_unmapped", {}),
                        "suggested": rep.get("suggested", {}), "unmapped": rep["unmapped"],
                        "from_library": rep["library_real"]}
    # KiCad renders become the inventory thumbnails (iso last = the one the cards show)
    for view in ("bottom", "top", "iso"):
        if view in renders:
            dst = os.path.join(out_dir, f"kicad_{view}.png")
            shutil.copyfile(renders[view], dst)
            res["files"][f"kicad_{view}"] = os.path.basename(dst)
    if renders.get("iso"):
        res["files"]["thumb"] = "kicad_iso.png"
    src_copy = os.path.join(out_dir, "source" + os.path.splitext(path)[1])
    shutil.copyfile(path, src_copy)
    res["files"]["source"] = os.path.basename(src_copy)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    return res


def add_model(conn, oid, src, license=None, source_url=None, source_site=None, note=None, on_start=None,
              recolor=None):
    """Full pipeline in one tracked run; returns the normalize result + registered asset ids.
    on_start(run_id) is called once the run exists (the dashboard polls it)."""
    o = db.get_object(conn, oid)
    if o is None:
        raise KeyError(oid)
    with track.run(conn, "model", provider="blender", notes=f"model for {oid}: {src}"[:200]) as r:
        if on_start:
            on_start(r.id)
        path, sha, url = fetch(oid, src)
        source_url = source_url or url
        from .kicad import board_colors
        colors = board_colors(o.get("colors"))
        out_dir = os.path.join(config.data_dir(), "model_cache",
                               build_key(conn, sha, colors, o["dims_mm"], path, recolor))
        cached = _load_cached(out_dir)
        if cached:
            with track.step("cache_hit", object_id=oid, inputs={"key": os.path.basename(out_dir)}) as s:
                res = cached
                s.set(note="identical source + settings + parts library: reused the earlier build")
                if isinstance(res.get("parts"), dict):  # boards: keep the Parts library's "used by" lists current
                    # (for plain meshes Blender's "parts" is just a mesh count)
                    _note_parts_use(conn, oid, res["parts"])
        else:
            res = _build(conn, o, oid, path, colors, out_dir, recolor)
        res = dict(res, files=dict(res["files"]))
        with track.step("register", object_id=oid) as s:
            meta = dict(source_url=source_url, source_site=source_site or (urllib.parse.urlparse(source_url).hostname
                                                                          if source_url else "local file"),
                        license=license or "unknown", sha256=sha)
            ids = {}
            views = [("thumb", f"kicad_{v}") for v in ("bottom", "top") if res["files"].get(f"kicad_{v}")]
            for kind, key in (("usd", "usd"), ("glb", "glb"), ("obj", "obj"), *views, ("thumb", "thumb")):
                if res["files"].get(key):
                    ids[kind] = db.add_asset(conn, oid, kind, track.rel(os.path.join(out_dir, res["files"][key])),
                                             actor="pipeline", scale_factor=res.get("scale_factor"),
                                             dims_mm=res.get("dims_mm"), tri_count=res.get("tri_count"), **meta)
            ids["source"] = db.add_asset(conn, oid, "source", track.rel(os.path.join(out_dir, res["files"]["source"])),
                                         actor="pipeline", **meta)
            cid = db.add_candidate(conn, oid, source_url=source_url or src, license=meta["license"],
                                   local_path=track.rel(out_dir),
                                   scores={"dims_fit": res.get("dims_fit"), "flags": res.get("flags")}, rank=1)
            conn.execute("UPDATE candidates SET chosen=(id=?) WHERE object_id=?", (cid, oid))
            ch = {"primary_asset_id": ids.get("glb")}
            if o["status"] in ("draft", "identified"):
                ch["status"] = "model_found"
            db.update_object(conn, oid, ch, actor="pipeline", reason=note or "3D model added", run_id=r.id)
            s.set(asset_ids=ids, candidate_id=cid)
        from . import sim
        if sim.enabled():  # physics + semantics layer over the new geometry (assets/<id>/sim.usda)
            res["sim"] = sim.safe_build(conn, oid)
        res.update(run_id=r.id, asset_ids=ids, out_dir=track.rel(out_dir))
    return res


def remove_models(conn, oid, reason="model removed"):
    """Drop the object's generated 3D assets (rows + files under assets/<oid>/) and un-choose candidates."""
    rows = db.rows(conn.execute("SELECT id, path FROM assets WHERE object_id=?", (oid,)))
    with db.tx(conn):
        conn.execute("UPDATE objects SET primary_asset_id=NULL WHERE id=?", (oid,))
        conn.execute("DELETE FROM assets WHERE object_id=?", (oid,))
        conn.execute("UPDATE candidates SET chosen=0 WHERE object_id=?", (oid,))
        db.log_event(conn, oid, "user", "models_removed", {"assets": len(rows)}, None, reason)
    o = db.get_object(conn, oid)
    if o and o["status"] in ("model_found", "verified"):
        db.update_object(conn, oid, {"status": "identified"}, actor="user", reason=reason)
    shutil.rmtree(os.path.join(config.data_dir(), "assets", oid), ignore_errors=True)
    return len(rows)


def rebuild(conn, oid):
    """Re-run an object's model from its stored source (e.g. after the parts library gained real parts)."""
    src = conn.execute("SELECT path, source_url, license FROM assets WHERE object_id=? AND kind='source' "
                       "ORDER BY id DESC LIMIT 1", (oid,)).fetchone()
    if src is None:
        raise ValueError(f"{oid} has no stored model source to rebuild from")
    keep = os.path.join(_quarantine(oid), "rebuild_source" + os.path.splitext(src[0])[1])
    shutil.copyfile(os.path.join(config.data_dir(), src[0]), keep)
    remove_models(conn, oid, reason="rebuild with updated parts library")
    o = db.get_object(conn, oid)
    recolor = next((h.get("recolor") for h in o.get("model_hints") or []
                    if h.get("recolor") and src[1] in (h.get("page"), h.get("url"))), None)
    return add_model(conn, oid, keep, license=src[2], source_url=src[1], note="rebuilt with updated parts library",
                     recolor=recolor)
