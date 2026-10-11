"""Internal 3D parts library: the components board builds (objsearch.kicad) can place, at what quality,
and what is still missing. It is the continuous-improvement loop for model accuracy:

  every board build  -> parts the KiCad library can't supply are recorded here (status missing/unmapped),
                        and an approximate body (from the footprint's fabrication outline) is generated so
                        the model is still complete-ish; the object's model gets a flag
  add a real STEP    -> `python -m objsearch parts add <key> <step|url> --license .. [--transform ..]` or the
                        dashboard Parts page; it's aligned once (stored transform) and used by every later build
  rebuild            -> `python -m objsearch parts rebuild` re-runs models that used approximations

Lookup order for a KiCad model reference: library real -> KiCad installed -> KiCad online (cached)
-> library approximate -> generate approximate now.
"""
import json
import os
import re
import shutil

from . import config, db, track

HEIGHT_HINTS_MM = [  # (regex on the KiCad model/footprint name, body height above the board) for approximations
    (r"USB_B", 10.9), (r"USB_A", 7.0), (r"USB_C", 3.3), (r"USB_Micro", 2.8), (r"USB_Mini", 4.0),
    (r"BarrelJack", 11.0), (r"PinSocket", 8.5), (r"PinHeader", 2.5), (r"Crystal_HC49", 3.6), (r"SW_|Button", 3.5),
    (r"DIP-", 3.3), (r"SOT-223", 1.8), (r"TS4\d|TACT|PUSH|RESET", 3.5), (r"CP_Elec", 5.4), (r"Relay", 15.5), (r"TerminalBlock", 10.0),
]
NON_PHYSICAL = re.compile(r"^(TP|TEST|FIDUC|FID|SJ|SOLDERJUMPER|JP-?SJ|FRAME|DUMMY|LOGO|MOUNT|HOLE|PAD$)", re.I)


def part_dir(*p):
    d = os.path.join(config.data_dir(), "parts3d", *p)
    os.makedirs(os.path.dirname(d) if p and "." in p[-1] else d, exist_ok=True)
    return d


def key_for(model_path):
    """'${KICAD10_3DMODEL_DIR}/Connector_USB.3dshapes/X.wrl' -> 'Connector_USB.3dshapes/X'."""
    rel = re.sub(r"^\$\{KICAD\d*_3DMODEL_DIR\}/", "", model_path.replace("\\", "/"))
    return os.path.splitext(rel)[0]


def get(conn, key):
    return db.row(conn.execute("SELECT * FROM parts3d WHERE key=?", (key,)).fetchone())


def upsert(conn, key, kind, status, object_id=None, **fields):
    """Create/update a library entry; never downgrades a real part; remembers which objects need it."""
    old = get(conn, key)
    seen = set((old or {}).get("seen_in") or [])
    if object_id:
        seen.add(object_id)
    if old and old["status"] == "real" and status != "real":
        status, fields = "real", {}
    now = db.now()
    vals = dict(kind=kind, status=status, seen_in=sorted(seen), updated_at=now, **fields)
    with db.tx(conn):
        if old is None:
            db._insert(conn, "parts3d", dict(key=key, created_at=now, **vals))
        else:
            sets = ", ".join(f"{k}=?" for k in vals)
            conn.execute(f"UPDATE parts3d SET {sets} WHERE key=?",
                         [*(db.enc(v) if k in db.JSON_COLS else v for k, v in vals.items()), key])
    return get(conn, key)


def library_path(conn, key, statuses=("real",)):
    p = get(conn, key)
    if p and p["status"] in statuses and p["path"]:
        full = os.path.join(config.data_dir(), p["path"])
        if os.path.exists(full):
            return full
    return None


# ---------------------------------------------------------------- geometry (gmsh / OpenCASCADE)

def _gmsh():
    import gmsh
    if not gmsh.isInitialized():
        gmsh.initialize(interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.clear()
    return gmsh


def _bbox(g):
    xs = g.model.getBoundingBox(-1, -1)
    return [round(xs[3] - xs[0], 3), round(xs[4] - xs[1], 3), round(xs[5] - xs[2], 3)], xs


def transform_step(src, dst, matrix):
    """Apply a 3x4 row-major affine matrix (mm) to every solid in src and write dst (STEP)."""
    g = _gmsh()
    ents = g.model.occ.importShapes(src)
    flat = [v for row in matrix for v in row] + [0, 0, 0, 1]
    g.model.occ.affineTransform(ents, flat)
    g.model.occ.synchronize()
    dims, _ = _bbox(g)
    g.write(dst)
    return dims


def approx_body(dst, xmin, xmax, ymin, ymax, height):
    """Box body in KiCad 3D model coordinates (mm, y up = -footprint y), sitting on the board (z 0..height)."""
    g = _gmsh()
    g.model.occ.addBox(xmin, ymin, 0, xmax - xmin, ymax - ymin, height)
    g.model.occ.synchronize()
    dims, _ = _bbox(g)
    g.write(dst)
    return dims


def fab_outline(footprint_text, layers=("F.Fab", "F.CrtYd")):
    """Outline bbox of a footprint (first layer in `layers` that has graphics), footprint coords (y down)."""
    from . import kicad
    fp = kicad.parse(footprint_text)[0]
    for layer in layers:
        pts = []
        for kind in ("fp_line", "fp_rect", "fp_poly", "fp_arc"):
            for item in kicad.children(fp, kind):
                lay = kicad.child(item, "layer")
                if lay is None or lay[1] != layer:
                    continue
                for k in ("start", "end", "mid"):
                    c = kicad.child(item, k)
                    if c is not None:
                        pts.append(kicad._nums(c)[:2])
                pts_node = kicad.child(item, "pts")
                if pts_node is not None:
                    pts += [kicad._nums(xy)[:2] for xy in kicad.children(pts_node, "xy")]
        if pts:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            return min(xs), max(xs), min(ys), max(ys), layer
    return None


def height_hint(name):
    for pat, h in HEIGHT_HINTS_MM:
        if re.search(pat, name, re.I):
            return h
    return 2.0


def approximate_for(conn, key, footprint_libref, object_id=None):
    """Generate (or reuse) an approximate body for a KiCad model reference from its footprint outline."""
    from . import kicad
    hit = library_path(conn, key, ("approximate",))
    if hit:
        upsert(conn, key, "kicad_model", "approximate", object_id)
        return hit
    lib, name = footprint_libref.split(":", 1)
    with open(kicad.share("footprints", f"{lib}.pretty", f"{name}.kicad_mod"), encoding="utf-8") as f:
        box = fab_outline(f.read())
    if box is None:
        upsert(conn, key, "kicad_model", "missing", object_id, notes="no footprint outline to approximate from")
        return None
    x0, x1, y0, y1, layer = box
    h = height_hint(key)
    dst = part_dir("approx", key.replace("/", "__") + ".step")
    dims = approx_body(dst, x0, x1, -y1, -y0, h)
    upsert(conn, key, "kicad_model", "approximate", object_id, path=track.rel(dst), dims_mm=dims,
           transform={"footprint": footprint_libref},
           notes=f"box from {layer} outline of {footprint_libref}, height {h} mm (estimate); add a real STEP")
    return dst


def add_real(conn, key, src, license, source_url=None, matrix=None, notes=None, footprint=None):
    """Store a real STEP for a part and mark it real. Alignment: matrix if given, else (with footprint, a KiCad
    'Lib:Name') auto-aligned to the footprint outline and oriented so its leads land on the pads."""
    footprint = footprint or footprint_for(conn, key)
    if matrix is None and footprint:
        matrix, scores = orient_by_pads(src, auto_align(src, footprint, all_candidates=True), footprint)
        notes = (notes + "; " if notes else "") + f"auto-aligned to {footprint} (lead-to-pad {min(scores)} mm)"
    dst = part_dir("real", key.replace("/", "__") + ".step")
    if matrix:
        dims = transform_step(src, dst, matrix)
    else:
        shutil.copyfile(src, dst)
        g = _gmsh()
        g.model.occ.importShapes(dst)
        g.model.occ.synchronize()
        dims, _ = _bbox(g)
    return upsert(conn, key, "kicad_model", "real", path=track.rel(dst), license=license, source_url=source_url,
                  transform=matrix, dims_mm=dims, notes=notes)


def approximate_unmapped(conn, footprint_name, fp_text, value="", object_id=None):
    """Approximate body for a board footprint with no KiCad match, from the board's own drawing of it
    (fab/silkscreen/courtyard outline, else pad extents). Coordinates are footprint-local, so the model is
    placed with zero offset/rotation. -> STEP path or None."""
    from . import kicad
    key = f"eagle:{footprint_name}"
    hit = library_path(conn, key, ("real", "approximate"))
    if hit:
        p = get(conn, key)
        upsert(conn, key, p["kind"], p["status"], object_id)  # remember this object uses it too
        return hit, p["status"]
    box = fab_outline(fp_text, ("F.Fab", "F.SilkS", "F.CrtYd"))
    if box is None:
        pts = [p for v in kicad._pads(kicad.parse(fp_text)[0]).values() for p in v]
        if not pts:
            return None, None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        box = (min(xs) - 0.5, max(xs) + 0.5, min(ys) - 0.5, max(ys) + 0.5, "pads")
    x0, x1, y0, y1, layer = box
    h = height_hint(f"{footprint_name} {value}")
    dst = part_dir("approx", f"eagle__{footprint_name}.step")
    dims = approx_body(dst, x0, x1, -y1, -y0, h)
    fp_file = part_dir("footprints", f"eagle__{footprint_name}.kicad_mod")
    with open(fp_file, "w", encoding="utf-8") as f:
        f.write(fp_text)
    upsert(conn, key, "footprint", "approximate", object_id, path=track.rel(dst), dims_mm=dims,
           transform={"footprint": track.rel(fp_file)},
           notes=f"no KiCad footprint fits; box from the board's {layer} outline, height {h} mm (estimate)")
    return dst, "approximate"


def record_unmapped(conn, footprint, refs, object_id=None):
    return upsert(conn, f"eagle:{footprint}", "footprint", "unmapped", object_id,
                  notes=f"no KiCad footprint mapping; refs {', '.join(sorted(refs))[:200]}")


def listing(conn):
    return db.rows(conn.execute("SELECT * FROM parts3d ORDER BY CASE status WHEN 'missing' THEN 0 "
                                "WHEN 'unmapped' THEN 1 WHEN 'approximate' THEN 2 ELSE 3 END, key"))


def affected_objects(conn):
    """Objects whose current model used an approximation or lacks parts (worth rebuilding when the library grows)."""
    ids = set()
    for p in listing(conn):
        ids.update(p.get("seen_in") or [])
    return sorted(ids)


# ---------------------------------------------------------------- auto-alignment of a real STEP to a footprint

def _rotations():
    """The 24 proper axis-aligned rotations as 3x3 integer matrices."""
    import itertools
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = [[0] * 3 for _ in range(3)]
            for r, c in enumerate(perm):
                m[r][c] = signs[r]
            det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
            if det == 1:
                out.append(m)
    return out


def footprint_text(ref):
    """'Lib:Name' (installed KiCad library) or a data-relative/absolute .kicad_mod path -> file text."""
    from . import kicad
    if ref.endswith(".kicad_mod"):
        path = ref if os.path.isabs(ref) else os.path.join(config.data_dir(), ref)
    else:
        lib, name = ref.split(":", 1)
        path = kicad.share("footprints", f"{lib}.pretty", f"{name}.kicad_mod")
    with open(path, encoding="utf-8") as f:
        return f.read()


def footprint_for(conn, key):
    """The footprint a library part is aligned to (for auto-alignment of an uploaded STEP), or None."""
    p = get(conn, key)
    t = (p or {}).get("transform")
    return t.get("footprint") if isinstance(t, dict) else None


def auto_align(src, footprint_libref, height=None, all_candidates=False):
    """Guess the matrix that puts a vendor STEP onto a KiCad footprint: the largest solid (the body) is turned
    so its footprint matches the footprint's F.Fab outline (and its height the expected height), centred on the
    outline with its underside on the board. 180-degree ambiguity remains -> check the render (or flip())."""
    from . import kicad
    g = _gmsh()
    ents = g.model.occ.importShapes(src)
    g.model.occ.synchronize()
    vols = [t for t in ents if t[0] == 3] or ents
    boxes = [g.model.getBoundingBox(*t) for t in vols]
    body = max(boxes, key=lambda b: (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]))
    ext = [body[3] - body[0], body[4] - body[1], body[5] - body[2]]
    ctr = [(body[0] + body[3]) / 2, (body[1] + body[4]) / 2, (body[2] + body[5]) / 2]
    x0, x1, y0, y1, _ = fab_outline(footprint_text(footprint_libref), ("F.Fab", "F.SilkS", "F.CrtYd"))
    want = [x1 - x0, y1 - y0, height or height_hint(footprint_libref)]
    scored = []
    for m in _rotations():
        rext = [abs(sum(m[r][c] * ext[c] for c in range(3))) for r in range(3)]
        scored.append((abs(rext[0] - want[0]) + abs(rext[1] - want[1]) + 0.5 * abs(rext[2] - want[2]), m, rext))
    best = min(sc for sc, _, _ in scored)
    out = []
    for sc, m, rext in scored:  # every rotation tied on size: up/down and front/back are told apart by the leads
        if sc > best + 1e-6:
            continue
        rc = [sum(m[r][c] * ctr[c] for c in range(3)) for r in range(3)]
        target = [(x0 + x1) / 2, -(y0 + y1) / 2, rext[2] / 2]  # 3D y is up (= -footprint y); underside at z=0
        out.append([[m[r][0], m[r][1], m[r][2], round(target[r] - rc[r], 4)] for r in range(3)])
    return out if all_candidates else out[0]


def flip(matrix, cx, cy):
    """Turn an aligned part 180 degrees about the vertical axis through (cx, cy) (fixes a backwards guess)."""
    r = [[-matrix[0][j] for j in range(3)], [-matrix[1][j] for j in range(3)], list(matrix[2][:3])]
    t = [2 * cx - matrix[0][3], 2 * cy - matrix[1][3], matrix[2][3]]
    return [r[i] + [round(t[i], 4)] for i in range(3)]


def _pin_points(src, matrix):
    """xy centres (3D coords) of the lower ends of non-body solids after alignment: where leads meet the board."""
    g = _gmsh()
    ents = g.model.occ.importShapes(src)
    flat = [v for row in matrix for v in row] + [0, 0, 0, 1]
    g.model.occ.affineTransform(ents, flat)
    g.model.occ.synchronize()
    vols = [t for t in ents if t[0] == 3]
    boxes = [g.model.getBoundingBox(*t) for t in vols]
    if len(boxes) < 2:
        return []
    body = max(boxes, key=lambda b: (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]))
    return [((b[0] + b[3]) / 2, (b[1] + b[4]) / 2) for b in boxes if b is not body and b[2] < 0.5]


def orient_by_pads(src, matrix, footprint_libref):
    """Choose the orientation whose leads land closest to the footprint pads. matrix: one matrix (compared with
    its 180-degree flip) or a list of candidates (e.g. auto_align(..., all_candidates=True))."""
    import math
    from . import kicad
    pads = [(x, -y) for v in kicad._pads(kicad.parse(footprint_text(footprint_libref))[0]).values() for x, y in v]
    options = matrix if isinstance(matrix[0][0], list) else [matrix, flip(matrix, matrix[0][3], matrix[1][3])]

    def score(m):
        pts = _pin_points(src, m)
        if not pts or not pads:
            return math.inf
        return sum(min(math.dist(p, q) for q in pads) for p in pts) / len(pts)
    scored = sorted(options, key=score)
    return scored[0], [round(score(m), 3) for m in options]


# ---------------------------------------------------------------- open CAD libraries (candidates for real STEPs)

OPEN_LIBRARIES = {
    "freecad": {"tree": "https://api.github.com/repos/FreeCAD/FreeCAD-library/git/trees/master?recursive=1",
                "raw": "https://raw.githubusercontent.com/FreeCAD/FreeCAD-library/master/",
                "page": "https://github.com/FreeCAD/FreeCAD-library/blob/master/",
                "license": "CC-BY-3.0 (FreeCAD Parts Library)"},
}
_GENERIC = {"horizontal", "vertical", "tht", "smd", "smt", "pcb", "jack", "connector", "type", "mm", "p2", "54mm",
            "female", "male", "socket", "header", "x", "h", "v", "the", "and", "of"}


_VARIANTS = {"micro", "mini", "smd", "smt", "a", "c", "right", "angle", "dual", "stacked"}


def _tokens(text):
    words = re.findall(r"[a-z]+\d*|\d+[a-z]*", text.lower().replace("_", " ").replace("-", " "))
    return [w for w in words if w not in _GENERIC and len(w) > 0]


def library_index(name="freecad", refresh=False):
    """Cached list of STEP paths in an open CAD library (one GitHub API call, then offline)."""
    import httpx
    lib = OPEN_LIBRARIES[name]
    path = os.path.join(config.data_dir(), "third_party", f"{name}_library_index.json")
    if os.path.exists(path) and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    r = httpx.get(lib["tree"], timeout=60, headers={"User-Agent": "objsearch/0.1", "Accept": "application/vnd.github+json"})
    r.raise_for_status()
    steps = [e["path"] for e in r.json().get("tree", []) if e["type"] == "blob" and
             e["path"].lower().endswith((".step", ".stp"))]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(steps, f)
    return steps


def search_open_libraries(key, notes="", limit=8):
    """Candidate real STEPs for a library part, ranked by shared name tokens (needs review: names are fuzzy)."""
    import urllib.parse
    want = set(_tokens(key.split("/")[-1].replace("eagle:", "") + " " + (notes or "")[:0]))
    out = []
    for name, lib in OPEN_LIBRARIES.items():
        for p in library_index(name):
            have = set(_tokens(os.path.splitext(p)[0]))
            shared = want & have
            if len(shared) >= 2 or (len(want) == 1 and shared):
                wrong_variant = (have - want) & _VARIANTS  # e.g. a 'micro'/'mini' USB when we want full-size
                out.append({"library": name, "path": p, "score": len(shared) - len(wrong_variant),
                            "shared": sorted(shared),
                            "url": lib["raw"] + urllib.parse.quote(p), "page": lib["page"] + urllib.parse.quote(p),
                            "license": lib["license"]})
    out.sort(key=lambda c: (-c["score"], len(c["path"])))
    return out[:limit]


def tint_step(src, dst, hex_color):
    """Give a colourless STEP a colour by appending AP214 presentation entities (STYLED_ITEM -> COLOUR_RGB) for
    every solid, so KiCad/OpenCASCADE readers show it. Solids that already carry a colour are left alone."""
    import re as _re
    text = open(src, encoding="latin-1").read()
    flat = _re.sub(r"\s+", " ", text)
    solids = _re.findall(r"#(\d+)\s*=\s*(?:MANIFOLD_SOLID_BREP|BREP_WITH_VOIDS|SHELL_BASED_SURFACE_MODEL)\s*\(", flat)
    styled = set(_re.findall(r"STYLED_ITEM\s*\([^;]*?,\s*#(\d+)\s*\)\s*;", flat))
    todo = [x for x in solids if x not in styled]
    ctx = _re.search(r"ADVANCED_BREP_SHAPE_REPRESENTATION\s*\(.*?,\s*#(\d+)\s*\)\s*;", flat) or \
        _re.search(r"#(\d+)\s*=\s*\(\s*GEOMETRIC_REPRESENTATION_CONTEXT", flat)
    if not todo or ctx is None:
        if src != dst:
            import shutil as _sh
            _sh.copyfile(src, dst)
        return {"tinted": 0, "already_coloured": len(styled)}
    n = max(int(x) for x in _re.findall(r"#(\d+)\s*=", flat)) + 1
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    ents, items = [], []
    for sid in todo:
        c, fc, fa, sf, ss, su, pa, si = range(n, n + 8)
        ents += [f"#{c}=COLOUR_RGB('',{r:.6f},{g:.6f},{b:.6f});", f"#{fc}=FILL_AREA_STYLE_COLOUR('',#{c});",
                 f"#{fa}=FILL_AREA_STYLE('',(#{fc}));", f"#{sf}=SURFACE_STYLE_FILL_AREA(#{fa});",
                 f"#{ss}=SURFACE_SIDE_STYLE('',(#{sf}));", f"#{su}=SURFACE_STYLE_USAGE(.BOTH.,#{ss});",
                 f"#{pa}=PRESENTATION_STYLE_ASSIGNMENT((#{su}));", f"#{si}=STYLED_ITEM('color',(#{pa}),#{sid});"]
        items.append(f"#{si}")
        n += 8
    ents.append(f"#{n}=MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION('',({','.join(items)}),#{ctx.group(1)});")
    i = text.rfind("ENDSEC;")  # end of DATA
    out = text[:i] + "\n".join(ents) + "\n" + text[i:]
    with open(dst, "w", encoding="latin-1") as f:
        f.write(out)
    return {"tinted": len(todo), "already_coloured": len(styled)}
