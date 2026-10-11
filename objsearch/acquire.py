"""Automatic 3D model sourcing for a catalog object (runs right after identification, in the background).

Close replicas of the exact item come first. Every candidate gets a TIER and the best tier wins:

  1 exact board layout   the product's own PCB design (.kicad_pcb/.brd from GitHub, known hints) built with KiCad;
                         LEGO: the official LDraw part
  2 detailed exact STEP  KiCad / FreeCAD / GitHub / research-agent STEP with >= 2000 triangles or >= 12 parts
  3 KiCad replica        catalog/boards/<id>.yaml -> objsearch.boardgen (real measurements + real KiCad parts)
  4 simple STEP          low-detail STEP that fits (kept as a fallback, never preferred over 1-3)
  5 photo box            box at the object's size with its photo on top (always builds, no network)

Fast sources run in tier order; a tier 1-2 hit is built at once. Otherwise the best of tier 3-5 is built right away
(so the inventory shows something within seconds) and the research agent then looks for a tier 1-2 source and
replaces the model only with a better tier. Known (learned) sources of tier 1-3 are used immediately.
Every successful source is written back to model_hints with its tier.

    python -m objsearch acquire arduino-uno-r3 [--no-agent] [--force]
"""
import glob
import logging
import os
import re

from . import config, db, models, track

log = logging.getLogger("objsearch.acquire")
KICAD_LICENSE = "CC-BY-SA-4.0 (KiCad 3D library, with exception)"
ELECTRONIC = {"module", "board", "sensor", "actuator", "camera", "storage", "power", "prototyping"}

AGENT_SYSTEM = """You find the closest possible 3D replica of ONE physical product (exact model/revision) so it can be
imported into Blender/USD. Priority, best first:
  (a) the product's own PCB design files (KiCad .kicad_pcb, Eagle .brd, Altium/zip) - an exact board layout;
  (b) a DETAILED STEP of the exact product (many parts/triangles);
  (c) only if nothing better exists: a simple STEP that fits.
Tools: github_board_search (design files) first, then kicad_find / open_library_search / github_cad_search with
the product's distinctive words (chip or module name, MPN, board marking), then web_search + web_fetch for
manufacturer CAD or design-file downloads. A library hit can be tested directly: "kicad:<model name>" or its URL.
try_model downloads, converts and measures a candidate; it reports "detail": board / detailed / simple. The run
stops by itself at the first good-fitting board or detailed model; simple fits are kept as a fallback, so keep
looking for something better while you have budget. A close variant of the same product (for
example another maker's version of the same module) is acceptable if its size fits; say so in the reason.
Report the license exactly as the source states it (or 'unknown'). Never invent URLs."""

ACQUIRE_RESULT = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "direct file URL that try_model accepted"},
        "license": {"type": "string", "description": "license as stated by the source, or 'unknown'"},
        "source_page": {"type": "string", "description": "page where the file is offered"},
        "reason": {"type": "string", "description": "why this candidate (fit, detail, official?)"},
    },
    "required": ["url", "license", "reason"],
}
TRY_MODEL = {"type": "object", "properties": {"url": {"type": "string", "description": "direct file URL"}},
             "required": ["url"]}


# ---------------------------------------------------------------- strict name matching

def _tokens(text):
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def _covers(query_tokens, file_tokens):
    """Every file-name token must appear in the query (allowing joined neighbours, e.g. raspberry+pi)."""
    q = set(query_tokens) | {a + b for a, b in zip(query_tokens, query_tokens[1:])}
    return all(t in q for t in file_tokens)


def _distinctive(tok):
    """A token that actually identifies a product: letters+digits (esp32, sg90, 12e) or a long word (arduino)."""
    return bool(re.search(r"[a-z]", tok) and re.search(r"\d", tok)) or (tok.isalpha() and len(tok) >= 6)


def match_names(names, candidates):
    """candidates: list of (path, display_name). Returns the best path whose file name is fully explained by one
    of the object's names (>= 2 tokens, >= 1 distinctive), preferring the most specific match, or None."""
    best = None
    for name in names:
        q = _tokens(name)
        for path, disp in candidates:
            ft = _tokens(disp)
            if len(ft) >= 2 and any(_distinctive(t) for t in ft) and _covers(q, ft) and \
                    (best is None or len(ft) > best[0]):
                best = (len(ft), path)
    return best[1] if best else None


def _object_names(o):
    return [n for n in [o.get("mpn"), o["name"], *(o.get("aliases") or [])] if n]


# ---------------------------------------------------------------- strategies (each returns a source dict or None)

def _known(o):
    for h in o.get("model_hints") or []:
        if h.get("source") == "boardgen" and h.get("spec"):  # reconstructed KiCad board (objsearch.boardgen)
            from . import boardgen
            spec = os.path.join(config.ROOT, h["spec"])
            if os.path.exists(spec):
                out = os.path.join(config.data_dir(), "boardgen", o["id"], f"{o['id']}.kicad_pcb")
                boardgen.build_pcb(boardgen.load(spec), out)
                return {"src": out, "license": h.get("license"), "source_url": h["spec"], "how": "reconstructed board",
                        "flag": h.get("flag"), "tier": 3}
        if h.get("url"):
            ext = os.path.splitext(h["url"].split("?")[0])[1].lower()
            # untested legacy hints: only board files are trusted as tier 1; a STEP's detail is unknown -> fallback
            tier = h.get("tier") or (1 if ext in models.BOARD_EXT else 4)
            return {"src": h["url"], "license": h.get("license"), "source_url": h.get("page") or h["url"],
                    "how": "known source", "recolor": h.get("recolor"), "tier": tier}
        if h.get("source") == "kicad" and h.get("ref"):
            from . import kicad
            if kicad.available():
                p = kicad.share("3dmodels", *(h["ref"] + ".step").split("/"))
                if os.path.exists(p):
                    return {"src": p, "license": KICAD_LICENSE, "source_url": f"kicad:{h['ref']}", "how": "KiCad ref",
                            "tier": h.get("tier") or 2}
    return None


_KICAD_LIST = []


def _kicad(o):
    from . import kicad
    if o.get("category") not in ELECTRONIC or not kicad.available():
        return None
    if not _KICAD_LIST:
        _KICAD_LIST.extend((p, os.path.splitext(os.path.basename(p))[0])
                           for p in glob.glob(kicad.share("3dmodels", "*.3dshapes", "*.step")))
    hit = match_names(_object_names(o), _KICAD_LIST)
    if not hit:
        return None
    ref = os.path.relpath(hit, kicad.share("3dmodels")).replace("\\", "/")
    return {"src": hit, "license": KICAD_LICENSE, "source_url": f"kicad:{ref[:-5]}", "how": "KiCad library"}


def _open_lib(o):
    from . import parts
    if o.get("category") == "lego":  # generic 'lego brick' files aren't the exact part; LEGO needs LDraw parts
        return None
    try:
        index = parts.library_index("freecad")
    except Exception as e:  # offline etc.: just skip this source
        log.info(f"open library unavailable: {e}")
        return None
    cands = [(p, os.path.splitext(os.path.basename(p))[0]) for p in index]
    hit = match_names(_object_names(o), cands)
    if not hit:
        return None
    import urllib.parse
    lib = parts.OPEN_LIBRARIES["freecad"]
    return {"src": lib["raw"] + urllib.parse.quote(hit), "license": lib["license"],
            "source_url": lib["page"] + urllib.parse.quote(hit), "how": "FreeCAD library"}


def size_fit(extents, dims_mm, tol=0.15):
    """Compare a model's extents (raw units) with the catalog size. The footprint (two largest sides) must match
    within tol; height may differ (pin headers, connectors and mounting parts vary between makers)."""
    want = sorted(dims_mm)
    fits = []
    for unit, f in (("m", 1000), ("mm", 1), ("in", 25.4), ("cm", 10)):
        got = sorted(x * f for x in extents)
        foot = max(abs(a - b) / b for a, b in zip(got[1:], want[1:]))
        fits.append((foot, unit, got))
    err, unit, got = min(fits)
    h = got[0] / want[0] if want[0] > 0 else 1.0
    out = {"size_mm": [round(x, 1) for x in got], "units_guess": unit, "size_error": f"{err:.0%}",
           "verdict": "good fit" if err <= tol and 0.4 <= h <= 3.0 else "size mismatch"}
    if not 0.8 <= h <= 1.25:
        out["height_note"] = f"height {got[0]:.1f} mm vs catalog {want[0]:.1f} mm (pins/headers?)"
    return out


DETAILED_TRIS, DETAILED_PARTS = 2000, 12
TIER_NAMES = {1: "exact board layout", 2: "detailed STEP", 3: "KiCad replica", 4: "simple STEP", 5: "photo box"}


def detail(r):
    """try_model result -> 'detailed' | 'simple' (few triangles and few parts) | 'box' (< 100 triangles)."""
    tris = r.get("triangles")
    if tris is None:
        return "detailed"
    if tris < 100:
        return "box"
    return "detailed" if tris >= DETAILED_TRIS or (r.get("parts") or 0) >= DETAILED_PARTS else "simple"


def tier_of(r):
    """Tier of a measured candidate (see module doc)."""
    d = r.get("detail") or detail(r)
    return {"board": 1, "detailed": 2, "simple": 4}.get(d, 5)


def try_model(url, o):
    """Agent tool: download (quarantine), convert quickly and measure vs the catalog size. No Blender."""
    import cascadio
    import trimesh
    if url.startswith("kicad:") or (".3dshapes/" in url and not url.startswith("http")):
        from . import kicad
        ref = url.removeprefix("kicad:").removesuffix(".step")
        url = kicad.share("3dmodels", *(ref + ".step").split("/"))
        if not os.path.exists(url):
            return {"ok": False, "error": f"no KiCad model named {ref}"}
    path, sha, _ = models.fetch(o["id"], url)
    ext = os.path.splitext(path)[1].lower()
    if ext in models.BOARD_EXT:
        from . import kicad
        mesh, _info = kicad.board_to_step(path, os.path.splitext(path)[0] + "_try", renders=(), fmt="glb",
                                          object_id=o["id"])
    elif ext in (".step", ".stp"):
        mesh = os.path.splitext(path)[0] + ".glb"
        if cascadio.step_to_glb(path, mesh, tol_linear=0.1) != 0:
            return {"ok": False, "error": "STEP conversion failed"}
    else:
        mesh = path
    m = trimesh.load(mesh, force="scene")
    ext_raw = sorted(float(x) for x in m.extents)
    tris = sum(len(g.faces) for g in m.geometry.values()) if hasattr(m, "geometry") else 0
    nparts = len(m.geometry) if hasattr(m, "geometry") else 1
    out = {"ok": True, "format": ext, "triangles": tris, "parts": nparts, "sha256": sha[:12]}
    if o.get("dims_mm") and max(ext_raw) > 0:
        out.update(size_fit(ext_raw, o["dims_mm"]))
    out["detail"] = "board" if ext in models.BOARD_EXT else detail(out)
    if out["detail"] == "box":
        out["warning"] = "very low detail: probably a placeholder box"
    elif out["detail"] == "simple":
        out["note"] = f"simple model ({tris} triangles, {nparts} parts): kept only as a fallback"
    return out


def _agent(conn, o, backend):
    from .llm import tools
    from .llm.base import Budget, StopAgent, Tool
    from .sources import web
    handlers = {"web_fetch": lambda a: web.fetch_text(a["url"]),
                "kicad_find": lambda a: __import__("objsearch.kicad", fromlist=["find"]).find(a["query"]),
                "catalog_lookup": tools.catalog_lookup_handler(conn)}
    if hasattr(backend, "grounded_search"):
        handlers["web_search"] = lambda a: backend.grounded_search(a["query"])
    from . import parts
    tried = []

    meta = {}  # url -> {license, page} from search results, so an early stop keeps the license

    def tool_try(a):
        out = try_model(a["url"], o)
        tried.append((a["url"], out))
        if _ok(out) and tier_of(out) <= 2:
            raise StopAgent(f"good fit ({out.get('detail') or detail(out)}): {a['url']}")  # board / detailed: stop
        return out

    def tool_open_lib(a):
        hits = parts.search_open_libraries(a["query"], limit=6)
        meta.update({h["url"]: {"license": h["license"], "page": h["page"]} for h in hits})
        return {"results": [{"name": h["path"].split("/")[-1], "url": h["url"], "license": h["license"],
                             "page": h["page"]} for h in hits]}
    from .sources import github

    def tool_github_boards(a):
        hits = github.search_cad(a["query"], limit=8, exts=github.BOARD_EXT)
        meta.update({h["url"]: {"license": h["license"], "page": h["page"]} for h in hits})
        return {"results": hits}

    def tool_github(a):
        hits = github.search_cad(a["query"], limit=8)
        meta.update({h["url"]: {"license": h["license"], "page": h["page"]} for h in hits})
        return {"results": hits}
    tlist = tools.build_tools(handlers, ("kicad_find", "web_search", "web_fetch", "catalog_lookup"))
    if github.available():
        tlist.append(Tool("github_cad_search", "Search GitHub for CAD files (.step/.stp) by product words. Returns "
                          "DIRECT download URLs (for try_model) with each repository's license.",
                          {"type": "object", "properties": {"query": {"type": "string",
                                                                       "description": "e.g. 'SSD1306 0.96 OLED'"}},
                           "required": ["query"]}, tool_github))
        tlist.append(Tool("github_board_search", "Search GitHub for PCB DESIGN files (.kicad_pcb/.brd) by product "
                          "words or board marking. Returns DIRECT download URLs (for try_model) with each repo's "
                          "license. The product's own board files are the best possible source.",
                          {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
                          tool_github_boards))
    tlist.append(Tool("open_library_search", "Search open CAD libraries (FreeCAD Parts Library) for STEP files by "
                      "name words. Results can go straight to try_model.",
                      {"type": "object", "properties": {"query": {"type": "string", "description": "product words"}},
                       "required": ["query"]}, tool_open_lib))
    tlist.append(Tool("try_model", "Test a candidate: a direct file URL (STEP/zip/board/mesh) or 'kicad:<model>' from "
                      "kicad_find. Downloads, converts and measures it against the product's size.", TRY_MODEL,
                      tool_try))
    dims = f"{o['dims_mm']} mm (L x W x H)" if o.get("dims_mm") else "unknown"
    content = [f"Product: {o['name']}\nManufacturer: {o.get('manufacturer') or 'unknown'}\nMPN: {o.get('mpn') or '-'}\n"
               f"Also called: {', '.join(o.get('aliases') or []) or '-'}\nCategory: {o.get('category')}\nSize: {dims}"]
    b = Budget(max_llm_calls=10, max_tool_calls=16, max_searches=4)
    res = backend.run(AGENT_SYSTEM, content, tlist, ACQUIRE_RESULT, b)
    out = res.output
    if not out:  # stopped at a good fit (or ran out of budget): take the best candidate it tested
        good = [(u, r) for u, r in tried if _ok(r)]
        if not good:
            return None
        u, r = min(good, key=lambda x: (tier_of(x[1]), float(str(x[1].get("size_error", "100%")).rstrip("%"))))
        m = meta.get(u, {})
        out = {"url": u, "license": m.get("license", "unknown"), "source_page": m.get("page"),
               "reason": f"tested good fit ({r.get('size_error')} footprint error)"}
    url = out["url"]
    # tier from what try_model measured; an untested submission only counts as a simple fallback
    tier = min((tier_of(r) for u, r in tried if u == url and _ok(r)), default=4)
    if url.startswith("kicad:") or (".3dshapes/" in url and not url.startswith("http")):
        from . import kicad
        ref = url.removeprefix("kicad:").removesuffix(".step")
        return {"src": kicad.share("3dmodels", *(ref + ".step").split("/")), "license": KICAD_LICENSE,
                "source_url": f"kicad:{ref}", "how": "research agent (KiCad library)", "reason": out.get("reason"),
                "tier": tier}
    return {"src": out["url"], "license": out.get("license") or "unknown",
            "source_url": out.get("source_page") or out["url"], "how": "research agent", "reason": out.get("reason"),
            "tier": tier}


def _ok(r):
    return bool(r.get("ok") and r.get("verdict") == "good fit" and "warning" not in r)


def _ranked_github(o, exts, limit=20):
    """GitHub hits whose FILE NAME carries the product's key token, best-named first (content matches of host
    boards that merely use the part are excluded this way)."""
    from .sources import github
    names = _object_names(o)
    words = {t for n in names for t in _tokens(n)}
    key = [t for n in names[:2] for t in _tokens(n) if _distinctive(t)]
    if not key:
        return []
    hits, seen = [], set()
    # MPN, then the key token alone (catches shorter file names, e.g. "GMT020-02 ..." for "GMT020-02-8P"),
    # then the full name / first alias - 3 queries max (GitHub code search allows 10 requests a minute)
    queries = [names[0], key[0]] + [n for n in names[1:] if n]
    for q in list(dict.fromkeys(queries))[:3]:
        try:
            found = github.search_cad(q, limit=limit, exts=exts)
        except RuntimeError as e:  # one bad query (or a rate limit) must not drop the other
            log.info(f"github search {q!r}: {e}")
            continue
        for h in found:
            if h["url"] not in seen:
                seen.add(h["url"])
                hits.append(h)
    ranked = []
    for h in hits:
        ft = _tokens(os.path.splitext(h["name"])[0])
        if any(k in t for k in key for t in ft):
            ranked.append((-sum(any(w == t or (len(w) > 1 and w in t) for t in ft) for w in words), len(ft), h))
    ranked.sort(key=lambda x: x[:2])
    return [h for _, _, h in ranked]


def _best_tested(o, hits, tries, how):
    """try_model the first few hits; return the best-tier good fit (stopping early on tier 1-2)."""
    best = None
    for h in hits[:tries]:
        try:
            r = try_model(h["url"], o)
        except Exception as e:  # bad download / not really CAD: next one
            log.info(f"{how} candidate {h['name']} failed: {e}")
            continue
        if not _ok(r):
            continue
        cand = {"src": h["url"], "license": h["license"], "source_url": h["page"], "how": how, "fit": r,
                "tier": tier_of(r)}
        if best is None or cand["tier"] < best["tier"]:
            best = cand
        if best["tier"] <= 2:
            break
    return best


def _github_boards(o, tries=2):
    """Tier 1: the product's own PCB design files on GitHub (file named after the product), built with KiCad."""
    from . import kicad
    from .sources import github
    if o.get("category") == "lego" or not o.get("dims_mm") or not github.available() or not kicad.available():
        return None
    return _best_tested(o, _ranked_github(o, github.BOARD_EXT), tries, "GitHub board design")


def _github(o, tries=3):
    """GitHub STEP files named after the product: tier 2 if detailed, tier 4 if simple."""
    from .sources import github
    if o.get("category") == "lego" or not o.get("dims_mm") or not github.available():
        return None
    return _best_tested(o, _ranked_github(o, github.CAD_EXT), tries, "GitHub CAD search")


def _boardgen(o):
    """Tier 3: a KiCad replica spec exists for this object (catalog/boards/<id>.yaml)."""
    from . import boardgen, kicad
    spec = os.path.join(config.ROOT, "catalog", "boards", f"{o['id']}.yaml")
    if not os.path.exists(spec) or not kicad.available():
        return None
    sp = boardgen.load(spec)
    d = os.path.join(config.data_dir(), "boardgen", o["id"])
    if sp.get("overlay") or sp.get("extras"):  # board + real-STEP parts merged into one GLB
        out, _ = boardgen.build_model(sp, d)
    else:
        out = boardgen.build_pcb(sp, os.path.join(d, f"{o['id']}.kicad_pcb"))["path"]
    return {"src": out, "license": sp.get("license", "reconstructed board (objsearch boardgen)"),
            "source_url": os.path.relpath(spec, config.ROOT).replace("\\", "/"), "how": "KiCad replica",
            "flag": sp.get("flag"), "tier": 3, "recolor": sp.get("recolor")}


def refresh_placeholder(conn, oid, backend=None):
    """Rebuild an object's placeholder box (new photo, crop or size) without re-running the source search.
    Only touches objects whose current model IS a placeholder."""
    cur = conn.execute("SELECT license FROM candidates WHERE object_id=? AND chosen=1 ORDER BY id DESC LIMIT 1",
                       (oid,)).fetchone()
    if not cur or cur[0] != "generated":
        raise ValueError(f"{oid}: current model is not a placeholder")
    o = db.get_object(conn, oid)
    with track.run(conn, "acquire", provider="pipeline", notes=f"refresh placeholder {oid}"):
        old = [x[0] for x in conn.execute("SELECT id FROM assets WHERE object_id=?", (oid,))]
        cand = _placeholder(conn, o, backend)
        res = models.add_model(conn, oid, cand["src"], license="generated", note="placeholder refreshed")
        _retire(conn, oid, old, "placeholder refreshed")
        _add_flag(conn, oid, cand["flag"])
    return res


def _ldraw(o):
    """LEGO: the exact part geometry from the LDraw library (official parts, CC BY 4.0), in the catalog color."""
    part = o.get("ldraw_id") or next((h["ref"] for h in o.get("model_hints") or []
                                      if h.get("source") == "ldraw" and h.get("ref")), None)
    if not part:
        return None
    from .sources import ldraw
    part = str(part).lower().removesuffix(".dat")
    dst = os.path.join(config.data_dir(), "ldraw", "models", f"{part}_{'-'.join(o.get('colors') or ['red'])}.glb")
    info = ldraw.part_glb(part, dst, o.get("colors"))
    return {"src": dst, "license": ldraw.LICENSE, "source_url": f"https://library.ldraw.org/parts/list?tableSearch={part}",
            "how": "LDraw library", "detail": f"{info['triangles']} triangles, color {info['color_name']}"}


def _placeholder(conn, o, backend=None):
    """Always builds: a box at the object's size, with the user's photo (or the reference image) on top. With an
    LLM available the photo is first cropped to the object (one vision call), so the top shows the item."""
    from . import fallback, physical
    if not o.get("dims_mm"):
        return None
    photo = physical.best_photo(conn, o["id"])
    if photo and backend is not None:
        try:
            crop = os.path.join(config.data_dir(), "placeholders", f"{o['id']}_crop.jpg")
            photo = physical.crop_to_object(photo, o["name"], backend, crop) or photo
        except Exception as e:  # quota / no key: the uncropped photo still works
            log.info(f"{o['id']}: photo crop skipped: {e}")
    dst = os.path.join(config.data_dir(), "placeholders", f"{o['id']}.glb")
    fallback.photo_box(o["dims_mm"], dst, photo)
    size = {"catalog": "catalog size", "datasheet": "datasheet size", "measured": "measured size",
            "photo_estimate": "size estimated from the photo", "category_default": "TYPICAL size for its category"
            }.get(o.get("dims_source") or "catalog", "catalog size")
    return {"src": dst, "license": "generated", "source_url": None, "how": "placeholder",
            "flag": f"placeholder model: a box at the {size}" + (" with your photo on top" if photo else "") +
                    " (no real 3D model found yet)"}


# ---------------------------------------------------------------- orchestration

def auto_enabled():
    env = os.environ.get("OBJSEARCH_AUTO_ACQUIRE")
    if env is not None:
        return env not in ("0", "false", "no", "")
    return bool(config.load().get("acquire", {}).get("auto", True))


def acquire(conn, oid, use_agent=True, backend=None, force=False):
    """Find and build the closest replica for oid (tiers: see module doc).
    Returns {"how", "tier", "result", "tried", "run_id", "placeholder_first"}."""
    o = db.get_object(conn, oid)
    if o is None:
        raise KeyError(oid)
    has_model = conn.execute("SELECT 1 FROM assets WHERE object_id=? AND kind='glb'", (oid,)).fetchone()
    if has_model and not force:
        return {"how": "already has a model", "result": None, "tried": []}
    tried = []
    with track.run(conn, "acquire", provider="pipeline", notes=f"acquire {oid}") as r:
        try:  # LLM for the size estimate / photo crop; only when LLM use is allowed (use_agent)
            est = (backend or _default_backend()) if use_agent else None
        except Exception:  # no key: category default size, uncropped photo
            est = None
        if not o.get("dims_mm"):  # every object gets a size first, so the placeholder can always be built
            from . import physical
            physical.ensure_dims(conn, oid, est)
            o = db.get_object(conn, oid)

        def find(name, fn):
            """Run one source (tracked step) -> candidate with a tier, or None."""
            with track.step(f"source_{name}", object_id=oid) as s:
                try:
                    cand = fn()
                except track.QuotaExceeded as e:
                    s.skip(f"quota: {e}")
                except RuntimeError as e:  # e.g. no LLM key: move on to the next source
                    s.skip(str(e)[:200])
                except Exception as e:  # a broken source must never end the ladder
                    log.warning(f"{oid}: source {name} failed: {type(e).__name__}: {e}")
                    s.skip(f"error: {type(e).__name__}: {str(e)[:180]}")
                if not cand:
                    s.skip("no match")
                cand.setdefault("tier", 4)
                s.set(src=str(cand["src"])[:300], how=cand["how"], tier=cand["tier"],
                      tier_name=TIER_NAMES.get(cand["tier"]))
                return cand
            return None

        def build(cand, name):
            """Build a candidate as the object's model (replacing what it had). -> result or None."""
            with track.step(f"build_tier{cand['tier']}", object_id=oid, inputs={"how": cand["how"]}) as s:
                tried.append(cand["how"])
                old_assets = [x[0] for x in conn.execute("SELECT id FROM assets WHERE object_id=?", (oid,))]
                try:
                    res = models.add_model(conn, oid, cand["src"], license=cand.get("license"),
                                           source_url=cand.get("source_url"), note=f"auto: {cand['how']}",
                                           **({"recolor": cand["recolor"]} if cand.get("recolor") else {}))
                except Exception as e:  # a bad candidate must not end the ladder
                    s.set(build_error=f"{type(e).__name__}: {str(e)[:300]}")
                    log.warning(f"{oid}: {cand['how']} candidate failed to build: {e}")
                    return None
                if old_assets:  # the new model built: retire what it replaces (placeholder / lower tier)
                    _retire(conn, oid, old_assets, f"replaced by {cand['how']} (tier {cand['tier']})")
                if cand.get("flag"):
                    _add_flag(conn, oid, cand["flag"])
                if cand["tier"] == 4:
                    _add_flag(conn, oid, "simple model: low-detail STEP used because no exact board, detailed STEP "
                                         "or KiCad replica exists yet")
                if str(cand.get("license") or "unknown").startswith("unknown"):
                    _add_flag(conn, oid, "license unknown: fine for a private inventory; check the source before "
                                         "sharing or redistributing the model")
                if name not in ("placeholder", "known", "ldraw", "boardgen"):  # ldraw/boardgen: the id is the hint
                    _learn_source(conn, oid, cand)
                s.set(model_run=res["run_id"], dims_mm=res.get("dims_mm"), flags=res.get("flags"))
                return {"how": cand["how"], "tier": cand["tier"], "result": res}
            return None

        # fast sources in tier order; known (learned) tier 1-3 and any tier 1-2 hit are built at once
        fallbacks = []
        for name, fn in (("known", lambda: _known(o)), ("ldraw", lambda: _ldraw(o)),
                         ("github_boards", lambda: _github_boards(o)), ("kicad", lambda: _kicad(o)),
                         ("open_lib", lambda: _open_lib(o)), ("github", lambda: _github(o)),
                         ("boardgen", lambda: _boardgen(o))):
            cand = find(name, fn)
            if cand is None:
                continue
            cand.setdefault("tier", 1 if name == "ldraw" else 2)
            if cand["tier"] <= 2 or (name == "known" and cand["tier"] <= 3):
                got = build(cand, name)
                if got:
                    return {**got, "tried": tried, "run_id": r.id, "placeholder_first": False}
            else:
                fallbacks.append((cand["tier"], len(fallbacks), name, cand))
        # nothing of tier 1-2 yet: build the best fallback now (tier 3 replica, 4 simple STEP, else 5 photo box)
        current = None
        for _, _, name, cand in sorted(fallbacks, key=lambda x: x[:2]):
            current = build(cand, name)
            if current:
                break
        if current is None:
            ph = find("placeholder", lambda: _placeholder(conn, o, est))
            if ph:
                ph["tier"] = 5
                current = build(ph, "placeholder")
        # research agent: only to beat what we have (tier 3 or worse) with an exact board or a detailed STEP
        if use_agent and (current is None or current["tier"] >= 3):
            cand = find("agent", lambda: _agent(conn, o, backend or _default_backend()))
            if cand and (current is None or cand["tier"] < current["tier"]):
                got = build(cand, "agent")
                if got:
                    current = got
        out = current or {"how": None, "tier": None, "result": None}
    return {**out, "tried": tried, "run_id": r.id,
            "placeholder_first": bool(out.get("tier") == 5 or ("placeholder" in tried and out.get("how") != "placeholder"))}


def _default_backend():
    from .identify import backend_status
    from .llm import get_backend
    if not backend_status()["ready"]:
        raise RuntimeError("no LLM key: research agent skipped")
    if config.load().get("provider", "gemini") == "gemini":
        from .llm.gemini_backend import GeminiBackend
        return GeminiBackend(thinking_level="LOW")  # tool-driven search: speed over deep reasoning
    return get_backend(config.load().get("provider"))


def _learn_source(conn, oid, cand):
    """Remember a working source on the object so the next acquire (e.g. after an inventory reset) is step 1."""
    o = db.get_object(conn, oid)
    hints = [h for h in (o.get("model_hints") or []) if h.get("url") != cand["src"]]
    if os.path.exists(str(cand["src"])):  # local library file: store its KiCad ref instead of a path
        ref = (cand.get("source_url") or "").removeprefix("kicad:")
        hint = {"source": "kicad", "ref": ref, "verified": True, "tier": cand.get("tier")} if ref else None
    else:
        hint = {"source": cand["how"], "url": cand["src"], "license": cand.get("license"),
                "page": cand.get("source_url"), "verified": True, "tier": cand.get("tier")}
    if hint:
        db.update_object(conn, oid, {"model_hints": [hint] + hints}, actor="pipeline",
                         reason=f"learned model source ({cand['how']})")


def _retire(conn, oid, asset_ids, reason):
    """Swap models only after the replacement built: drop the old asset rows and un-choose older candidates."""
    newest = conn.execute("SELECT MAX(id) FROM candidates WHERE object_id=?", (oid,)).fetchone()[0]
    with db.tx(conn):
        conn.execute(f"DELETE FROM assets WHERE id IN ({','.join('?' * len(asset_ids))})", asset_ids)
        conn.execute("UPDATE candidates SET chosen=0 WHERE object_id=? AND id<?", (oid, newest))
        db.log_event(conn, oid, "pipeline", "model_replaced", {"assets": len(asset_ids)}, None, reason)


def _add_flag(conn, oid, flag):
    row = conn.execute("SELECT id, scores FROM candidates WHERE object_id=? AND chosen=1 ORDER BY id DESC LIMIT 1",
                       (oid,)).fetchone()
    if row:
        sc = db.row(conn.execute("SELECT scores FROM candidates WHERE id=?", (row[0],)).fetchone())["scores"] or {}
        sc["flags"] = [flag] + (sc.get("flags") or [])
        conn.execute("UPDATE candidates SET scores=? WHERE id=?", (db.enc(sc), row[0]))
