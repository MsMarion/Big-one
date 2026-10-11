"""KiCad integration (kicad-cli, KiCad 10): turn PCB design files into realistic 3D models.

    board (.kicad_pcb | Eagle .brd | Altium/PADS/... | .zip of those)
      -> kicad-cli pcb import            (non-KiCad formats)
      -> sanitize                        (Eagle items on undefined layers block the exporter)
      -> attach KiCad 3D models          (imported footprints have none: each part is matched to the official
                                          KiCad footprint, pads are fitted (rotation + offset, least squares)
                                          and that footprint's 3D model is placed exactly)
      -> kicad-cli pcb export step       (board + copper pads + silkscreen + soldermask + parts)
      -> objsearch.models pipeline       (STEP -> GLB -> Blender normalize -> USD/GLB/OBJ + render)

Also: find(query) searches the ~7k installed KiCad 3D models (the agent's kicad_find tool).
Models missing from the local install are fetched from the official kicad-packages3D repo and cached.
"""
import glob
import json
import itertools
import math
import os
import re
import subprocess
import time
import urllib.parse
import zipfile

import httpx

from . import config, track

PACKAGES3D_RAW = "https://gitlab.com/kicad/libraries/kicad-packages3D/-/raw/master/"
BOARD_EXT = {".kicad_pcb", ".brd", ".pcbdoc", ".pcb", ".cpa", ".zip"}

# Eagle footprint name -> KiCad footprint(s) whose pads + 3D model to use; an entry may be (libref, {"z": mm}). Two-terminal chips, ICs and
# connectors common on maker boards; extend as new boards come in (unmapped parts are just left bare).
EAGLE_TO_KICAD = {
    "DIL28-3": ["Package_DIP:DIP-28_W7.62mm_Socket", ("Package_DIP:DIP-28_W7.62mm", {"z": 3.0})],  # chip seated in socket
    "1X06": ["Connector_PinSocket_2.54mm:PinSocket_1x06_P2.54mm_Vertical"],
    "1X08": ["Connector_PinSocket_2.54mm:PinSocket_1x08_P2.54mm_Vertical"],
    "1X10": ["Connector_PinSocket_2.54mm:PinSocket_1x10_P2.54mm_Vertical"],
    "2X03": ["Connector_PinHeader_2.54mm:PinHeader_2x03_P2.54mm_Vertical"],
    "PN61729": ["Connector_USB:USB_B_OST_USB-B1HSxx_Horizontal"],
    "POWERSUPPLY_DC-21MM": ["Connector_BarrelJack:BarrelJack_Horizontal"],
    "QS": ["Crystal:Crystal_HC49-4H_Vertical"],
    "RESONATOR": ["Crystal:Resonator_SMD_Murata_CSTxExxV-3Pin_3.0x1.1mm"],
    "PANASONIC_D": ["Capacitor_SMD:CP_Elec_6.3x5.4"],
    "SOT223": ["Package_TO_SOT_SMD:SOT-223-3_TabPin2"],
    "MLF32": ["Package_DFN_QFN:QFN-32-1EP_5x5mm_P0.5mm_EP3.45x3.45mm"],
    "MSOP08": ["Package_SO:MSOP-8_3x3mm_P0.65mm"],
    "SOT23-DBV": ["Package_TO_SOT_SMD:SOT-23-5"],
    "SOT-23": ["Package_TO_SOT_SMD:SOT-23"],
    "C0603-ROUND": ["Capacitor_SMD:C_0603_1608Metric"],
    "CT_CN0603": ["Capacitor_SMD:C_0603_1608Metric"],
    "R0603-ROUND": ["Resistor_SMD:R_0603_1608Metric"],
    "0805": ["Capacitor_SMD:C_0805_2012Metric"],
    "CHIP-LED0805": ["LED_SMD:LED_0805_2012Metric"],
    "CAY16": ["Resistor_SMD:R_Array_Convex_4x0603"],
    "MINIMELF": ["Diode_SMD:D_MiniMELF"],
    "SMB": ["Diode_SMD:D_SMB"],
    "L1812": ["Inductor_SMD:L_1812_4532Metric"],
}


# ---------------------------------------------------------------- locating KiCad

def root():
    env = os.environ.get("KICAD_DIR")
    if env and os.path.isdir(env):
        return env
    cands = sorted(glob.glob(r"C:\Program Files\KiCad\*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p)])
    for c in reversed(cands):
        if os.path.exists(os.path.join(c, "bin", "kicad-cli.exe")):
            return c
    return None


def cli():
    r = root()
    return os.path.join(r, "bin", "kicad-cli.exe") if r else None


def available():
    return cli() is not None


def share(*parts):
    return os.path.join(root(), "share", "kicad", *parts)


def run_cli(args, timeout=600):
    exe = cli()
    if not exe:
        raise RuntimeError("KiCad (kicad-cli) not found; install KiCad 10 or set KICAD_DIR")
    t0 = time.perf_counter()
    p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
    track.log.info(f"kicad-cli {args[0]} {args[1] if len(args) > 1 else ''} -> {p.returncode} "
                   f"({round((time.perf_counter() - t0) * 1e3)} ms)")
    out = (p.stdout + p.stderr).strip()
    if p.returncode != 0:
        raise RuntimeError(f"kicad-cli {' '.join(args[:3])} failed: {out[-800:]}")
    return out


# ---------------------------------------------------------------- minimal s-expression reader (with spans)

class Node(list):
    start = end = 0


class Str(str):
    """A quoted string atom (bare atoms stay plain str)."""


def parse(text):
    stack, i, n = [Node()], 0, len(text)
    while i < n:
        c = text[i]
        if c == "(":
            node = Node()
            node.start = i
            stack.append(node)
            i += 1
        elif c == ")":
            node = stack.pop()
            node.end = i + 1
            stack[-1].append(node)
            i += 1
        elif c == '"':
            j, buf = i + 1, []
            while text[j] != '"':
                if text[j] == "\\":
                    j += 1
                buf.append(text[j])
                j += 1
            stack[-1].append(Str("".join(buf)))
            i = j + 1
        elif c.isspace():
            i += 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '()"':
                j += 1
            stack[-1].append(text[i:j])
            i = j
    return stack[0]


def head(node):
    return node[0] if isinstance(node, Node) and node and isinstance(node[0], str) else None


def children(node, name):
    return [c for c in node if head(c) == name]


def child(node, name):
    c = children(node, name)
    return c[0] if c else None


def _nums(node):
    return [float(x) for x in node[1:] if isinstance(x, str) and re.fullmatch(r"-?[\d.]+(e-?\d+)?", x)]


# ---------------------------------------------------------------- sanitize

def sanitize(text):
    """Drop zones on undefined layers (Eagle keepouts) and park other undefined-layer items on Cmts.User."""
    tree = parse(text)
    cuts = []
    for top in tree:
        for node in top if isinstance(top, Node) else []:
            if head(node) == "zone":
                lay = child(node, "layer") or child(node, "layers")
                if lay is not None and "UNDEFINED" in lay[1:]:
                    cuts.append((node.start, node.end))
    for s, e in sorted(cuts, reverse=True):
        text = text[:s] + text[e:]
    return text.replace('(layer "UNDEFINED")', '(layer "Cmts.User")'), len(cuts)


# ---------------------------------------------------------------- KiCad footprint library + 3D models

def _pads(fp):
    out = {}
    for p in children(fp, "pad"):
        at = child(p, "at")
        if at is not None and len(p) > 1:
            xy = _nums(at)[:2]
            out.setdefault(str(p[1]), []).append((xy[0], xy[1]))
    return out


def load_footprint(libref):
    lib, name = libref.split(":", 1)
    path = share("footprints", f"{lib}.pretty", f"{name}.kicad_mod")
    with open(path, encoding="utf-8") as f:
        fp = parse(f.read())[0]
    models = []
    for m in children(fp, "model"):
        off = _nums(child(child(m, "offset"), "xyz")) if child(m, "offset") is not None else [0, 0, 0]
        rot = _nums(child(child(m, "rotate"), "xyz")) if child(m, "rotate") is not None else [0, 0, 0]
        models.append((str(m[1]), off, rot))
    return {"pads": _pads(fp), "models": models, "name": libref}


def resolve_model(model_path, cache_dir=None):
    """${KICADn_3DMODEL_DIR}/Lib.3dshapes/x.wrl|step -> local .step (installed library, else download+cache)."""
    rel = re.sub(r"^\$\{KICAD\d*_3DMODEL_DIR\}/", "", model_path.replace("\\", "/"))
    rel = os.path.splitext(rel)[0] + ".step"
    local = share("3dmodels", *rel.split("/"))
    if os.path.exists(local):
        return local
    cache_dir = cache_dir or os.path.join(config.data_dir(), "third_party", "kicad3d")
    cached = os.path.join(cache_dir, *rel.split("/"))
    if os.path.exists(cached):
        return cached
    url = PACKAGES3D_RAW + "/".join(urllib.parse.quote(p) for p in rel.split("/"))
    t0 = time.perf_counter()
    r = httpx.get(url, timeout=120, follow_redirects=True, headers={"User-Agent": "objsearch/0.1"})
    track.record_http("GET", url, r.status_code, len(r.content), None, round((time.perf_counter() - t0) * 1e3))
    if r.status_code != 200 or not r.content.lstrip()[:20].startswith(b"ISO-10303-21"):
        return None
    os.makedirs(os.path.dirname(cached), exist_ok=True)
    with open(cached, "wb") as f:
        f.write(r.content)
    return cached


def find(query, limit=15):
    """Search installed KiCad 3D models by words in the file name (e.g. 'ESP32 WROOM', 'barrel jack')."""
    words = [w.lower() for w in re.findall(r"[\w.]+", query)]
    hits = []
    for p in glob.glob(share("3dmodels", "*.3dshapes", "*.step")):
        name = os.path.relpath(p, share("3dmodels")).replace("\\", "/")
        low = name.lower().replace("_", " ").replace("-", " ")
        score = sum(w in low or w in name.lower() for w in words)
        if score == len(words):
            hits.append(name)
    return {"models": sorted(hits, key=len)[:limit], "library": "KiCad packages3D (CC-BY-SA-4.0 with exception)"}


# ---------------------------------------------------------------- pad fitting + model placement

def fit(src, dst):
    """Rigid 2D fit mapping KiCad footprint pads (src) onto board footprint pads (dst), by pad name.
    Returns (theta_rad, (tx, ty), rms_mm, n_pairs) in KiCad footprint coordinates (y down)."""
    pairs = []
    for name, ps in src.items():
        if name in dst and len(ps) == len(dst[name]) == 1:
            pairs.append((ps[0], dst[name][0]))
    if len(pairs) < 2:
        return None
    cs = [sum(p[0][k] for p in pairs) / len(pairs) for k in (0, 1)]
    cd = [sum(p[1][k] for p in pairs) / len(pairs) for k in (0, 1)]
    num = den = 0.0
    for (sx, sy), (dx, dy) in pairs:
        ax, ay, bx, by = sx - cs[0], sy - cs[1], dx - cd[0], dy - cd[1]
        num += ax * by - ay * bx
        den += ax * bx + ay * by
    th = math.atan2(num, den)
    c, s = math.cos(th), math.sin(th)
    t = (cd[0] - (c * cs[0] - s * cs[1]), cd[1] - (s * cs[0] + c * cs[1]))
    err = [((c * sx - s * sy + t[0]) - dx) ** 2 + ((s * sx + c * sy + t[1]) - dy) ** 2 for (sx, sy), (dx, dy) in pairs]
    return th, t, math.sqrt(sum(err) / len(err)), len(pairs)


_CONV = (-1, 1)  # verified by render (Uno DIP socket / headers on their pads): (sign of model offset y vs footprint y, sign of model z-rotation vs footprint rotation)
PAD_ALIASES = {"C": "1", "K": "1", "A": "2", "+": "1", "-": "2"}  # Eagle diode/LED/polarized-cap pads -> KiCad


def fit_best(src, dst):
    """fit() trying, in order: same pad names, Eagle->KiCad polarity aliases, and (for <= 4 pads) every pin
    assignment. A rigid fit can't mirror, so a wrong assignment shows up as a large rms and loses."""
    tries = [dst, {PAD_ALIASES.get(k, k): v for k, v in dst.items()}]
    # small parts: permute the board pads that share names with the library pads (extra board pads, e.g.
    # Eagle's multi-hole slots, are ignored); near-symmetric pad triangles make the plain name match unreliable
    common = [n for n in dst if n in src]
    if len(src) <= 4 and len(common) == len(src) and all(len(src[n]) == len(dst[n]) == 1 for n in common):
        for perm in itertools.permutations(common):
            tries.append({k: dst[d] for k, d in zip(perm, common)})
    best = None
    for d in tries:
        f = fit(src, d)
        if f and (best is None or f[2] < best[2] - 1e-9):
            best = f
    return best


_PAD_RE = re.compile(r'\(pad\s+"([^"]*)"[^\n]*?\n?\s*\(at\s+(-?[\d.]+)\s+(-?[\d.]+)')


def footprint_index(rebuild=False):
    """{libref: {pad_name: [[x, y], ...]}} for every installed KiCad footprint (<= 40 pads), cached per KiCad
    version under data_dir/third_party (first build ~2-3 min, then instant)."""
    ver = os.path.basename(root() or "kicad")
    path = os.path.join(config.data_dir(), "third_party", f"kicad_fp_index_{ver}.json")
    if os.path.exists(path) and not rebuild:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    idx = {}
    for f in glob.glob(share("footprints", "*.pretty", "*.kicad_mod")):
        lib = os.path.basename(os.path.dirname(f))[:-len(".pretty")]
        with open(f, encoding="utf-8") as fh:
            pads = {}
            for name, x, y in _PAD_RE.findall(fh.read()):
                pads.setdefault(name, []).append([float(x), float(y)])
        if 1 < len(pads) <= 40:
            idx[f"{lib}:{os.path.basename(f)[:-len('.kicad_mod')]}"] = pads
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f)
    return idx


def fit_unlabeled(lib_pts, board_pts, tol=0.3):
    """Name-free rigid registration: every library pad must land on a distinct board pad within tol.
    Tries each pad pair correspondence with matching spacing. -> (theta, t, rms, n_matched) or None."""
    if len(lib_pts) < 2 or len(lib_pts) > len(board_pts) or len(lib_pts) > 12:
        return None
    best = None
    a0, a1 = lib_pts[0], next((p for p in lib_pts[1:] if math.dist(p, lib_pts[0]) > 0.5), None)
    if a1 is None:
        return None
    da = math.dist(a0, a1)
    for b0, b1 in itertools.permutations(board_pts, 2):
        if abs(math.dist(b0, b1) - da) > tol:
            continue
        th = math.atan2(b1[1] - b0[1], b1[0] - b0[0]) - math.atan2(a1[1] - a0[1], a1[0] - a0[0])
        c, s = math.cos(th), math.sin(th)
        t = (b0[0] - (c * a0[0] - s * a0[1]), b0[1] - (s * a0[0] + c * a0[1]))
        used, err = set(), []
        for x, y in lib_pts:
            px, py = c * x - s * y + t[0], s * x + c * y + t[1]
            j, d = min(((j, math.dist((px, py), q)) for j, q in enumerate(board_pts) if j not in used),
                       key=lambda z: z[1], default=(None, math.inf))
            if d > tol:
                break
            used.add(j)
            err.append(d * d)
        else:
            rms = math.sqrt(sum(err) / len(err))
            if best is None or rms < best[2]:
                best = (th, t, rms, len(err))
    return best


def suggest_footprint(board_pads, limit=5, max_rms=0.3, index=None):
    """Rank KiCad footprints by how well their pads fit this board footprint (rigid fit, pin permutations for
    small parts). -> [(libref, rms_mm, theta_deg), ...] best first."""
    idx = index if index is not None else footprint_index()
    bpts = [tuple(p) for v in board_pads.values() for p in v]
    out = []
    for libref, pads in idx.items():
        lpts = [tuple(p) for v in pads.values() for p in v]
        # same pad count, or the board has a few extra (e.g. Eagle slots built from several holes)
        if not (len(bpts) - 4 <= len(lpts) <= len(bpts)) or len(lpts) < 3:
            continue
        f = fit_unlabeled(lpts, bpts)
        if f and f[2] <= max_rms and f[3] >= max(3, math.ceil(0.75 * len(bpts))):
            out.append((libref, round(f[2], 3), round(math.degrees(f[0]), 1), f[3]))
    out.sort(key=lambda x: (-x[3], x[1]))  # most pads explained first, then tightest fit
    return [(a, b, c) for a, b, c, _ in out[:limit]]


def _model_sexpr(path, off, rot):
    p = path.replace("\\", "/")
    return (f'\n\t\t(model "{p}"\n\t\t\t(offset (xyz {off[0]:.4f} {off[1]:.4f} {off[2]:.4f}))\n'
            f'\t\t\t(scale (xyz 1 1 1))\n\t\t\t(rotate (xyz {rot[0]:.3f} {rot[1]:.3f} {rot[2]:.3f}))\n\t\t)')


def attach_models(text, mapping=EAGLE_TO_KICAD, max_rms=0.5, conn=None, object_id=None):
    """Insert 3D models into footprints that have none. Returns (new_text, report).
    With conn, the internal parts library (objsearch.parts) is used first, gaps get generated approximations,
    and missing/unmapped parts are recorded there (seen_in=object_id) so the library keeps improving."""
    from . import parts
    tree = parse(text)
    pcb = tree[0]
    inserts, libcache = [], {}
    report = {"placed": 0, "unmapped": {}, "ignored": {}, "poor_fit": [], "missing_model": [], "approximate": [],
              "library_real": []}
    for fp in children(pcb, "footprint"):
        if child(fp, "model") is not None:
            continue
        name = str(fp[1])
        ref = next((str(p[2]) for p in children(fp, "property") if len(p) > 2 and p[1] == "Reference"), "?")
        value = next((str(p[2]) for p in children(fp, "property") if len(p) > 2 and p[1] == "Value"), "")
        specs = mapping.get(name)
        if specs is None:
            not_fitted = re.search(r"(^|[\s-])(NM|DNP|DNF)$", value.strip(), re.I)
            if parts.NON_PHYSICAL.match(name) or not_fitted or not _pads(fp):
                report["ignored"].setdefault(name, []).append(ref)
                continue
            specs = _learned_or_suggested(name, _pads(fp), conn, object_id, report)
            if specs is None:  # nothing in KiCad fits: approximate body from the board's own outline
                if conn is not None:
                    path, status = parts.approximate_unmapped(conn, name, text[fp.start:fp.end], value, object_id)
                    if path:
                        inserts.append((fp.end - 1, _model_sexpr(path, (0, 0, 0), (0, 0, 0))))
                        report["placed"] += 1
                        key = "library_real" if status == "real" else "approximate_unmapped"
                        if status == "real":
                            report[key].append(f"eagle:{name}")
                        else:
                            report.setdefault(key, {}).setdefault(name, []).append(ref)
                        continue
                report["unmapped"].setdefault(name, []).append(ref)
                continue
        board_pads = _pads(fp)
        if child(fp, "layer") is not None and child(fp, "layer")[1] == "B.Cu":
            report["poor_fit"].append(f"{ref}: bottom side not supported yet")
            continue
        text_models = []
        for spec in specs:
            libref, extra = spec if isinstance(spec, tuple) else (spec, {})
            kfp = libcache.get(libref) or libcache.setdefault(libref, load_footprint(libref))
            f = fit_best(kfp["pads"], board_pads)
            if f is None or f[2] > max_rms:  # names may differ between libraries: try geometry alone
                f = fit_unlabeled([p for v in kfp["pads"].values() for p in v],
                                  [p for v in board_pads.values() for p in v]) or f
            if f is None or f[2] > max_rms:
                report["poor_fit"].append(f"{ref} ({name}->{libref}): " + ("too few matching pads" if f is None
                                                                          else f"rms {f[2]:.2f} mm"))
                continue
            th, t, _, _ = f
            ys, rs = _CONV
            c, s = math.cos(rs * th), math.sin(rs * th)
            for mpath, off, rot in kfp["models"]:
                local = _resolve(mpath, libref, conn, object_id, report)
                if local is None:
                    continue
                o = (c * off[0] - s * off[1] + t[0], s * off[0] + c * off[1] + ys * t[1], off[2] + extra.get("z", 0))
                text_models.append(_model_sexpr(local, o, (rot[0], rot[1], rot[2] + rs * math.degrees(th))))
                report["placed"] += 1
        if text_models:
            inserts.append((fp.end - 1, "".join(text_models)))
    for pos, s in sorted(inserts, reverse=True):
        text = text[:pos] + s + "\n\t" + text[pos:]
    if conn is not None:
        for name, refs in report["unmapped"].items():
            parts.record_unmapped(conn, name, refs, object_id)
    return text, report


def index_ready():
    ver = os.path.basename(root() or "kicad")
    return os.path.exists(os.path.join(config.data_dir(), "third_party", f"kicad_fp_index_{ver}.json"))


def _learned_or_suggested(name, board_pads, conn, object_id, report, max_rms=0.2):
    """A KiCad footprint for an unmapped board footprint: one learned earlier (parts library), else the best
    pad-geometry match from the footprint index (if built: `python -m objsearch parts index`). Learns it."""
    from . import parts
    key = f"eagle:{name}"
    if conn is not None:
        row = parts.get(conn, key)
        if row and isinstance(row.get("transform"), dict) and row["transform"].get("libref"):
            report.setdefault("learned", {})[name] = row["transform"]["libref"]
            return [row["transform"]["libref"]]
    if not index_ready():
        return None
    hits = suggest_footprint(board_pads, limit=1, max_rms=max_rms)
    if not hits:
        return None
    libref, rms, _ = hits[0]
    if not load_footprint(libref)["models"]:
        return None
    report.setdefault("suggested", {})[name] = libref
    if conn is not None:
        parts.upsert(conn, key, "footprint", "real", object_id, transform={"libref": libref, "rms": rms, "auto": True},
                     notes=f"auto-matched by pad geometry to {libref} (rms {rms} mm) - verify in the render")
    return [libref]


def _resolve(mpath, libref, conn, object_id, report):
    """Library real -> KiCad installed/online -> library approximate / generate one. Records what happened."""
    from . import parts
    key = parts.key_for(mpath)
    if conn is not None:
        real = parts.library_path(conn, key)
        if real:
            report["library_real"].append(key)
            parts.upsert(conn, key, "kicad_model", "real", object_id)
            return real
    local = resolve_model(mpath)
    if local:
        return local
    if conn is not None:
        approx = parts.approximate_for(conn, key, libref, object_id)
        if approx:
            report["approximate"].append(key)
            return approx
    report["missing_model"].append(mpath)
    return None


# ---------------------------------------------------------------- board colors + renders

MASK_HEX = {"teal": "#00979C", "blue": "#1C4E9C", "green": "#1E6B33", "red": "#B3201E", "black": "#151515",
            "white": "#EDEDED", "purple": "#5B2A86", "yellow": "#D4B21C", "orange": "#D2691E"}


def board_colors(color_names):
    """Catalog colors (e.g. ['teal', 'white']) -> (mask_hex, silk_hex) or None if no board color is named."""
    names = [c.lower() for c in color_names or []]
    mask = next((MASK_HEX[c] for c in names if c in MASK_HEX and c != "white"), None)
    if mask is None:
        return None
    return mask, ("#151515" if mask == MASK_HEX["white"] else "#F4F4F4")


def set_board_colors(text, mask_hex, silk_hex="#F4F4F4", finish="HAL SnPb", thickness=1.6):
    """Put a stackup with these solder-mask/silkscreen colors and copper finish (e.g. "ENIG" renders gold) into the
    board (replacing any existing one)."""
    m, k = mask_hex.lstrip("#").upper(), silk_hex.lstrip("#").upper()
    layers = [
        f'(layer "F.SilkS" (type "Top Silk Screen") (color "#{k}FF"))',
        '(layer "F.Paste" (type "Top Solder Paste"))',
        f'(layer "F.Mask" (type "Top Solder Mask") (color "#{m}E6") (thickness 0.01))',
        '(layer "F.Cu" (type "copper") (thickness 0.035))',
        f'(layer "dielectric 1" (type "core") (color "FR4 natural") (thickness {thickness - 0.09:.3f}) (material "FR4") '
        '(epsilon_r 4.5) (loss_tangent 0.02))',
        '(layer "B.Cu" (type "copper") (thickness 0.035))',
        f'(layer "B.Mask" (type "Bottom Solder Mask") (color "#{m}E6") (thickness 0.01))',
        '(layer "B.Paste" (type "Bottom Solder Paste"))',
        f'(layer "B.SilkS" (type "Bottom Silk Screen") (color "#{k}FF"))',
        f'(copper_finish "{finish}")',
        '(dielectric_constraints no)',
    ]
    stack = "(stackup\n" + "".join(f"\t\t\t{x}\n" for x in layers) + "\t\t)"
    setup = child(parse(text)[0], "setup")
    if setup is None:
        raise ValueError("board has no (setup) section")
    old = child(setup, "stackup")
    if old is not None:
        return text[:old.start] + stack + text[old.end:]
    i = setup.start + len("(setup")
    return text[:i] + "\n\t\t" + stack + text[i:]


RENDER_VIEWS = {"iso": ["--rotate", "-50,0,30", "--perspective", "--zoom", "1.05"],
                "iso_back": ["--rotate", "130,0,30", "--perspective", "--zoom", "1.05"],
                "top": ["--side", "top"], "bottom": ["--side", "bottom"]}


def render(pcb, out_png, view="iso", size=(1200, 800)):
    """KiCad's own 3D renderer (~3 s, board stackup colors) -> PNG."""
    run_cli(["pcb", "render", *RENDER_VIEWS[view], "--quality", "high", "--use-board-stackup-colors",
             "--background", "opaque", "-w", str(size[0]), "-h", str(size[1]), "-o", os.path.abspath(out_png),
             os.path.abspath(pcb)], timeout=300)
    return out_png


# ---------------------------------------------------------------- board -> STEP

def _extract_board(src, work):
    """zip -> the first board file inside (zip-slip safe); otherwise the file itself."""
    if not src.lower().endswith(".zip"):
        return src
    z = zipfile.ZipFile(src)
    names = [n for n in z.namelist() if not n.startswith("__MACOSX") and os.path.splitext(n)[1].lower() in BOARD_EXT - {".zip"}]
    if not names:
        raise ValueError("no PCB board file (.kicad_pcb/.brd/...) in the zip")
    names.sort(key=lambda n: (not n.lower().endswith(".kicad_pcb"), len(n)))
    dst = os.path.abspath(os.path.join(work, os.path.basename(names[0])))
    if not dst.startswith(os.path.abspath(work) + os.sep):
        raise ValueError("unsafe path in zip")
    with open(dst, "wb") as f:
        f.write(z.read(names[0]))
    return dst


def finish_renders(jobs):
    """Wait for parallel KiCad renders (as one tracked step) -> {view: png path}."""
    with track.step("kicad_render", inputs={"views": list(jobs)}) as s:
        out = {v: f.result() for v, f in jobs.items()}
        s.set(renders={v: track.rel(p) for v, p in out.items()}, note="rendered in parallel with export/convert")
    return out


def board_to_step(src, work, add_models=True, colors=None, renders=("iso", "top", "bottom"), object_id=None,
                  wait_renders=True, fmt="step"):
    """Any supported board file -> STEP with 3D parts (+ KiCad renders). Tracked as kicad_* steps.
    colors: (mask_hex, silk_hex) or None to keep the board's own. Returns (step_path, info)."""
    os.makedirs(work, exist_ok=True)
    info = {}
    board = _extract_board(src, work)
    pcb = os.path.join(work, "board.kicad_pcb")
    with track.step("kicad_import", inputs={"src": os.path.basename(board)}) as s:
        if board.lower().endswith(".kicad_pcb"):
            with open(board, encoding="utf-8") as f:
                text = f.read()
            s.set(imported=False)
        else:
            out = run_cli(["pcb", "import", "--format", "auto", "--report-format", "text", "-o", pcb, board])
            with open(pcb, encoding="utf-8") as f:
                text = f.read()
            stats = dict(re.findall(r"(Footprints|Tracks|Vias|Zones):\s*(\d+)", out))
            s.set(imported=True, **{k.lower(): int(v) for k, v in stats.items()})
            info["import"] = stats
        text, dropped = sanitize(text)
        s.set(dropped_undefined_zones=dropped)
    if add_models:
        with track.step("kicad_models", object_id=object_id) as s:
            run = track.current_run()
            text, rep = attach_models(text, conn=run.conn if run else None, object_id=object_id)
            s.set(placed=rep["placed"], library_real=rep["library_real"], approximate=rep["approximate"],
                  unmapped=rep["unmapped"], ignored={k: len(v) for k, v in rep["ignored"].items()},
                  poor_fit=rep["poor_fit"][:20], missing_model=rep["missing_model"][:20])
            info["models"] = rep
    if colors:
        text = set_board_colors(text, *colors)
        info["colors"] = colors
    fin = re.search(r'\(copper_finish "([^"]+)"\)', text)
    info["copper_finish"] = fin.group(1) if fin else None  # KiCad's GLB export ignores it; normalize applies it
    with open(pcb, "w", encoding="utf-8") as f:
        f.write(text)
    # renders only need the .kicad_pcb: run them in parallel with the STEP export (and, if the caller collects
    # them later via finish_renders(), with the STEP->mesh conversion too)
    jobs = {}
    if renders:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=len(renders))
        jobs = {v: pool.submit(render, pcb, os.path.join(work, f"render_{v}.png"), v) for v in renders}
        pool.shutdown(wait=False)
    step = os.path.join(work, f"board.{fmt}")
    with track.step("kicad_export", inputs={"format": fmt}) as s:
        # glb: ~5 s and goes straight to Blender; step: ~15 s + ~8 s STEP->mesh conversion (CAD-grade output)
        run_cli(["pcb", "export", fmt, "-f", "--subst-models", "--include-pads", "--include-silkscreen",
                 "--include-soldermask", "-o", step, pcb], timeout=1200)
        s.set(file=track.rel(step), bytes=os.path.getsize(step))
    if jobs and wait_renders:
        info["renders"] = finish_renders(jobs)
    elif jobs:
        info["render_jobs"] = jobs
    return step, info
