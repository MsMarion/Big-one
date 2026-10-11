"""Reference data for an object, after its 3D model: the datasheet (if one exists) and the physical details a
simulation needs (mass, materials, grip friction). Rough values are fine - they feed rigged sim/training data -
but each one says where it came from.

    python -m objsearch docs <ids> [--force]

Results: the PDF under <data>/docs/<id>/ with an evidence row (kind 'datasheet', local_path), the object's
`physics` JSON, and a rebuilt sim.usda (objsearch.sim) carrying the new mass/friction.
"""
import hashlib
import logging
import os
import re
import urllib.parse

from . import config, db, physical, track

log = logging.getLogger("objsearch.docs")

SYSTEM = """You collect reference data for ONE physical product, for a robot-simulation inventory.

1. Datasheet: find the product's datasheet PDF (manufacturer, or a big distributor/maker shop such as Mouser,
   Digi-Key, LCSC, Adafruit, SparkFun, Espressif, Raspberry Pi). Use web_search; when you have a DIRECT .pdf
   link, call grab_datasheet - it downloads and verifies it. You may also try a well-known datasheet URL you are
   confident about; grab_datasheet rejects anything that isn't a real PDF. One datasheet is enough. Generic
   items (LEGO, breadboards, flash drives) often have none - then skip this step.
2. Physical details: mass in grams (datasheet, vendor 'weight' field, or your best estimate), main materials by
   part (e.g. PCB: FR4; shield: tinplate; housing: ABS), surface feel and a static friction coefficient
   against a rubber gripper pad, restitution, and anything fragile for a gripper (pins, glass, flex cables).
Rough values are fine; say where each came from. Then call submit_result. Never invent URLs."""

RESULT = {
    "type": "object",
    "properties": {
        "datasheet_url": {"type": "string", "description": "the PDF grab_datasheet accepted, or '' if none"},
        "mass_g": {"type": "number", "description": "mass in grams"},
        "mass_source": {"type": "string", "enum": ["datasheet", "vendor page", "estimate"]},
        "materials": {"type": "array", "items": {
            "type": "object", "properties": {"part": {"type": "string"}, "material": {"type": "string"}},
            "required": ["part", "material"]}},
        "static_friction": {"type": "number", "description": "vs rubber gripper pad, typically 0.3-1.0"},
        "restitution": {"type": "number", "description": "bounciness 0-1, small parts ~0.05-0.3"},
        "fragile": {"type": "string", "description": "parts to avoid gripping, or 'none'"},
        "notes": {"type": "string"},
    },
    "required": ["datasheet_url", "mass_g", "mass_source", "materials", "static_friction", "restitution"],
}
GRAB = {"type": "object", "properties": {"url": {"type": "string", "description": "direct PDF URL"}},
        "required": ["url"]}
MAX_SEARCHES = 3
SKIP_CATEGORIES = {"lego"}  # no datasheets; category physics (ABS) are accurate enough


def auto_enabled():
    env = os.environ.get("OBJSEARCH_AUTO_DOCS")
    if env is not None:
        return env not in ("0", "false", "no", "")
    return bool(config.load().get("docs", {}).get("auto", True))


def _pdf_title(data, fallback):
    m = re.search(rb"/Title\s*\(([^)]{3,120})\)", data[:200_000])
    if m:
        try:
            t = m.group(1).decode("latin-1").strip()
            junk = ("microsoft word", "untitled", "localhost", "http", "file:", "c:\\")
            if t and not t.lower().startswith(junk) and not re.search(r"\.(html?|docx?|indd)$", t, re.I):
                return t
        except Exception:
            pass
    return fallback


def grab_datasheet(conn, oid, url):
    """Download a PDF (guarded), keep it under docs/<id>/ and record it as evidence. -> summary for the agent."""
    from .sources import web
    final, data, ctype = web.download(url)
    if not data.startswith(b"%PDF"):
        raise ValueError(f"not a PDF ({ctype or 'unknown type'}): pass a direct .pdf link")
    sha = hashlib.sha256(data).hexdigest()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(urllib.parse.urlparse(final).path))[:80] or "datasheet"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    rel = os.path.join("docs", oid, f"{sha[:10]}_{name}").replace("\\", "/")
    path = os.path.join(config.data_dir(), rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    title = _pdf_title(data, name)
    if not conn.execute("SELECT 1 FROM evidence WHERE object_id=? AND local_path=?", (oid, rel)).fetchone():
        db.add_evidence(conn, oid, final, title=title[:200], kind="datasheet", trust="downloaded PDF", local_path=rel)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    return {"ok": True, "saved": rel, "title": title, "pages": pages, "kb": len(data) // 1024}


def has_datasheet(conn, oid):
    return conn.execute("SELECT 1 FROM evidence WHERE object_id=? AND kind='datasheet' AND local_path IS NOT NULL",
                        (oid,)).fetchone() is not None


def research(conn, oid, backend=None, force=False):
    """Datasheet + physical details for oid; updates physics and rebuilds sim.usda. -> summary dict."""
    from .llm import tools
    from .llm.base import Budget, Tool
    from .sources import web
    o = db.get_object(conn, oid)
    if o is None:
        raise KeyError(oid)
    if (o.get("physics") or {}).get("researched_at") and not force:
        return {"skipped": "already researched"}
    if (o.get("category") or "") in SKIP_CATEGORIES:
        return {"skipped": f"no datasheets for {o['category']}; category physics used"}
    phys = None
    with track.run(conn, "docs", provider="pipeline", notes=f"docs {oid}") as r:
        with track.step("research_docs", object_id=oid) as s:
            if backend is None:
                from .acquire import _default_backend
                try:
                    backend = _default_backend()
                except RuntimeError as e:  # no LLM key: category physics stay in place
                    s.skip(str(e)[:200])
            grabbed = []

            def tool_grab(a):
                out = grab_datasheet(conn, oid, a["url"])
                grabbed.append(a["url"])
                return out
            searches = []

            def tool_search(a):  # soft limit: the agent is told to wrap up instead of being cut off mid-way
                if len(searches) >= MAX_SEARCHES:
                    return {"error": "search limit reached: call submit_result now with your best values"}
                searches.append(a["query"])
                return backend.grounded_search(a["query"])
            handlers = {"web_fetch": lambda a: web.fetch_text(a["url"])}
            if hasattr(backend, "grounded_search"):
                handlers["web_search"] = tool_search
            tlist = tools.build_tools(handlers, ("web_search", "web_fetch"))
            tlist.append(Tool("grab_datasheet", "Download a datasheet PDF by its direct URL and keep it with the "
                              "product. Rejects anything that is not a real PDF.", GRAB, tool_grab))
            dims = f"{o['dims_mm']} mm" if o.get("dims_mm") else "unknown"
            content = [f"Product: {o['name']}\nManufacturer: {o.get('manufacturer') or 'unknown'}\n"
                       f"MPN: {o.get('mpn') or '-'}\nAlso called: {', '.join(o.get('aliases') or []) or '-'}\n"
                       f"Category: {o.get('category')}\nSize: {dims}"
                       + ("\nA datasheet is already saved; only the physical details are needed."
                          if has_datasheet(conn, oid) else "")]
            res = backend.run(SYSTEM, content, tlist, RESULT, Budget(max_llm_calls=9, max_tool_calls=12,
                                                                     max_searches=MAX_SEARCHES + 3))
            out = res.output
            if not out:  # ran out of turns: one no-tools call for best estimates (datasheet kept if grabbed)
                s.set(agent_stop=res.stop_reason)
                res = backend.run(SYSTEM + "\nNo tools are available now: answer from what you know.",
                                  content + ["Give your best estimates now (datasheet_url: '' unless one was "
                                             "already saved)."], [], RESULT, Budget(2, 0, 0))
                out = res.output
            if not out:
                s.skip(f"no result ({res.stop_reason})")
            phys = {k: out[k] for k in ("mass_g", "mass_source", "materials", "static_friction", "restitution")
                    if out.get(k) is not None}
            phys = _sane(phys)
            phys.update(fragile=out.get("fragile") or "", notes=(out.get("notes") or "")[:500],
                        source=f"research agent ({out.get('mass_source', 'estimate')})", researched_at=db.now())
            db.update_object(conn, oid, {"physics": phys}, actor="agent", reason="datasheet / physical research",
                             run_id=r.id)
            s.set(datasheet=out.get("datasheet_url") or None, grabbed=grabbed, physics=phys)
        if phys is None:
            return {"run_id": r.id, "skipped": "research gave no result", "datasheet": has_datasheet(conn, oid)}
        from . import sim
        sim_res = sim.safe_build(conn, oid) if sim.enabled() else None
    return {"run_id": r.id, "datasheet": has_datasheet(conn, oid), "physics": phys, "sim": sim_res}


def _sane(p):
    """Clamp agent numbers to plausible ranges (a typo shouldn't make a 3 kg ESP32 in the simulator)."""
    if not (0.01 <= float(p.get("mass_g") or 0) <= 50_000):
        p.pop("mass_g", None)
        p.pop("mass_source", None)
    if "static_friction" in p:
        p["static_friction"] = min(max(float(p["static_friction"]), 0.05), 1.5)
        p["dynamic_friction"] = round(0.8 * p["static_friction"], 3)
    if "restitution" in p:
        p["restitution"] = min(max(float(p["restitution"]), 0.0), 0.95)
    return p


def safe_research(conn, oid, backend=None):
    """For the automatic pipeline: research never breaks identification or the model."""
    try:
        return research(conn, oid, backend)
    except Exception as e:
        log.warning(f"{oid}: docs research failed: {e}")
        return {"error": str(e)[:300]}


def physics_summary(o):
    """What the dashboard shows: final physics (researched or estimated)."""
    return physical.physics_for(o)
