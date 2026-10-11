"""Reconstruct a PCB as a KiCad board from a spec (catalog/boards/<id>.yaml) when no real design files exist.

Real parts go on it wherever they exist (KiCad library footprints/models, open CAD models); the board itself,
its holes, pads and silkscreen come from the manufacturer's documented dimensions and photos. KiCad then renders
real PCB materials (solder mask, ENIG/HASL copper, silkscreen), and the normal board pipeline (models.add_model)
turns it into GLB/USD like any downloaded KiCad project.

    python -m objsearch board <spec.yaml> [--build]
"""
import hashlib
import os
import re

import yaml

from . import config, kicad, track

MASK = {"black": "#151515", "green": "#1E6B33", "blue": "#1C4E9C", "red": "#B3201E", "white": "#EDEDED"}
SILK = {"white": "#F4F4F4", "black": "#151515", "yellow": "#E8D44D"}
FINISH = {"ENIG": "ENIG", "HASL": "HAL SnPb", "OSP": "OSP"}


def _f(x):
    return f"{float(x):.4f}".rstrip("0").rstrip(".")


def _fetch(url, spec_id):
    """Download a footprint/model once (guarded), cached under <data>/boardgen/<spec>/."""
    from .sources import web
    d = os.path.join(config.data_dir(), "boardgen", spec_id)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(url.split("?")[0]).replace("%20", "_"))
    path = os.path.join(d, hashlib.sha1(url.encode()).hexdigest()[:8] + "_" + name)
    if not os.path.exists(path):
        _, data, _ = web.download(url, max_bytes=30 * 2**20)
        os.makedirs(d, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    return path


def _model(path, offset=(0, 0, 0), rotate=(0, 0, 0)):
    p = path.replace("\\", "/")
    return (f'\t\t(model "{p}"\n\t\t\t(offset (xyz {" ".join(_f(v) for v in offset)}))\n'
            f'\t\t\t(scale (xyz 1 1 1))\n\t\t\t(rotate (xyz {" ".join(_f(v) for v in rotate)}))\n\t\t)\n')


def _props(ref, value, hide_ref=True):
    eff = "(effects (font (size 1 1) (thickness 0.15)))"
    return (f'\t\t(property "Reference" "{ref}" (at 0 0 0) (layer "F.SilkS"){" (hide yes)" if hide_ref else ""} {eff})\n'
            f'\t\t(property "Value" "{value}" (at 0 0 0) (layer "F.Fab") (hide yes) {eff})\n')


def _stroke(w):
    return f"(stroke (width {_f(w)}) (type solid))"


def _legacy_footprint(path):
    """KiCad 5 (module ...) footprint -> pads + silk lines/circles in current syntax (no model)."""
    root = kicad.parse(open(path, encoding="utf-8").read())[0]
    out = []
    for node in root[1:]:
        if not isinstance(node, list):
            continue
        h = kicad.head(node)
        get = lambda n, k: kicad.child(n, k)  # noqa: E731
        layer = get(node, "layer")
        layer = str(layer[1]) if layer is not None else ""
        width = get(node, "width")
        w = float(width[1]) if width is not None else 0.15
        if h == "fp_line" and layer.endswith("SilkS"):
            s, e = get(node, "start"), get(node, "end")
            out.append(f'\t\t(fp_line (start {_f(s[1])} {_f(s[2])}) (end {_f(e[1])} {_f(e[2])}) {_stroke(w)} '
                       f'(layer "{layer}"))\n')
        elif h == "fp_circle" and layer.endswith("SilkS"):
            c, e = get(node, "center"), get(node, "end")
            out.append(f'\t\t(fp_circle (center {_f(c[1])} {_f(c[2])}) (end {_f(e[1])} {_f(e[2])}) {_stroke(w)} '
                       f'(fill no) (layer "{layer}"))\n')
        elif h == "pad":
            num, typ, shape = str(node[1]), str(node[2]), str(node[3])
            at, size, drill = get(node, "at"), get(node, "size"), get(node, "drill")
            layers = " ".join(f'"{x}"' for x in get(node, "layers")[1:])
            out.append(f'\t\t(pad "{num}" {typ} {shape} (at {_f(at[1])} {_f(at[2])}) (size {_f(size[1])} {_f(size[2])})'
                       + (f" (drill {_f(drill[1])})" if drill is not None else "") + f" (layers {layers}))\n")
    return "".join(out)


def _library_footprint(libref, at, ref="J?"):
    """KiCad library footprint (same file format version as boards) placed at (x, y, rot); model paths resolved;
    its reference designator is hidden (a real board's silkscreen is what the spec draws)."""
    lib, name = libref.split(":", 1)
    text = open(kicad.share("footprints", f"{lib}.pretty", f"{name}.kicad_mod"), encoding="utf-8").read()
    text = re.sub(r'\(property "Reference" "REF\*\*"(.*?)\(layer "F\.SilkS"\)',
                  lambda m: f'(property "Reference" "{ref}"{m.group(1)}(layer "F.SilkS") (hide yes)', text, count=1,
                  flags=re.S)
    text = re.sub(r'^\s*\((version|generator|generator_version) [^\n]*\)\n', "", text, flags=re.M)
    text = text.replace(f'(footprint "{name}"', f'(footprint "{libref}"', 1)
    text = re.sub(r'\(layer "F.Cu"\)', f'(layer "F.Cu")\n\t\t(at {_f(at[0])} {_f(at[1])} {_f(at[2] if len(at) > 2 else 0)})',
                  text, count=1)
    text = re.sub(r"\$\{KICAD\d*_3DMODEL_DIR\}", kicad.share("3dmodels").replace("\\", "/"), text)
    return "\t" + text.strip().replace("\n", "\n\t") + "\n"


def _screw_head(path, d=3.8, h=1.3, colour="#B5B5B5"):
    """M2 pan-head screw head as seen on the board (no model exists in the KiCad library): a short cylinder with
    a cross recess, tinted steel. KiCad model coordinates: sits on the board surface, z up."""
    if os.path.exists(path):
        return path
    from . import parts
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g = parts._gmsh()
    occ = g.model.occ
    head = occ.addCylinder(0, 0, 0, 0, 0, h, d / 2)
    slot1 = occ.addBox(-d * 0.32, -0.25, h - 0.5, d * 0.64, 0.5, 0.6)
    slot2 = occ.addBox(-0.25, -d * 0.32, h - 0.5, 0.5, d * 0.64, 0.6)
    occ.cut([(3, head)], [(3, slot1), (3, slot2)])
    occ.synchronize()
    raw = path + ".raw.step"
    g.write(raw)
    parts.tint_step(raw, path, colour)
    os.remove(raw)
    return path


def _flip_to_back(pcb_path, flips):
    """Move footprints to the bottom side with KiCad's own pcbnew (see objsearch/kicad_flip.py)."""
    import json
    import subprocess
    py = os.path.join(kicad.root(), "bin", "python.exe")
    if not os.path.exists(py):
        py = os.path.join(kicad.root(), "bin", "python3")
    cfg = os.path.splitext(pcb_path)[0] + "_flips.json"
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(flips, f)
    p = subprocess.run([py, os.path.join(config.ROOT, "objsearch", "kicad_flip.py"), pcb_path, cfg],
                       capture_output=True, text=True, timeout=300)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("OBJSEARCH_RESULT ")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"KiCad flip failed: {(p.stderr or p.stdout)[-800:]}")
    res = json.loads(lines[-1][len("OBJSEARCH_RESULT "):])
    if res["missing"]:
        raise RuntimeError(f"footprints not found for flipping: {res['missing']}")
    return res


def build_pcb(spec, out_path):
    """Spec dict -> .kicad_pcb file. Returns {"path", "parts": [...], "sources": [...]}."""
    sid = spec["id"]
    b = spec["board"]
    w, h = b["size"]
    fps, sources = [], []
    # mounting holes: plated (the gold rings); name starts with MountingHole -> no 3D part is generated
    hl = spec.get("holes") or {}
    for i, (x, y) in enumerate(hl.get("at", [])):
        pad = (f'\t\t(pad "" np_thru_hole circle (at 0 0) (size {_f(hl["drill"])} {_f(hl["drill"])}) '
               f'(drill {_f(hl["drill"])}) (layers "*.Cu" "*.Mask"))\n') if not hl.get("pad") else (
               f'\t\t(pad "1" thru_hole circle (at 0 0) (size {_f(hl["pad"])} {_f(hl["pad"])}) '
               f'(drill {_f(hl["drill"])}) (layers "*.Cu" "*.Mask"))\n')
        fps.append(f'\t(footprint "MountingHole_{_f(hl["drill"])}mm{"_Pad" if hl.get("pad") else ""}" (layer "F.Cu") '
                   f'(at {_f(x)} {_f(y)})\n' + _props(f"H{i + 1}", "MountingHole")
                   + '\t\t(attr exclude_from_pos_files exclude_from_bom)\n' + pad + '\t)\n')
    for p in spec.get("parts") or []:
        fp = p["footprint"]
        at = p.get("at", [0, 0, 0])
        if "kicad" in fp:
            fps.append(_library_footprint(fp["kicad"], at, p["ref"]))
            sources.append({"part": p["name"], "footprint": f"KiCad library {fp['kicad']}"})
            continue
        body = _legacy_footprint(_fetch(fp["url"], sid))
        models = []
        for m in p.get("models") or []:
            path = _fetch(m["url"], sid)
            if m.get("tint"):
                from . import parts
                tinted = os.path.splitext(path)[0] + f"_tint{m['tint'].lstrip('#')}.step"
                parts.tint_step(path, tinted, m["tint"])
                path = tinted
            models.append(_model(path, m.get("offset", (0, 0, 0)), m.get("rotate", (0, 0, 0))))
            sources.append({"part": p["name"], "model": m["url"], "license": m.get("license"),
                            "tinted": m.get("tint")})
        fps.append(f'\t(footprint "objsearch:{re.sub(r"[^A-Za-z0-9_-]", "_", p["name"])[:40]}" (layer "F.Cu") '
                   f'(at {_f(at[0])} {_f(at[1])} {_f(at[2] if len(at) > 2 else 0)})\n'
                   + _props(p["ref"], p["name"]) + body + "".join(models) + "\t)\n")
    # back side: parts read from a photo of the back; written on the front, then flipped by KiCad itself
    flips = {}
    cal = spec.get("back_calibration") or {}
    k, (ox, oy) = cal.get("scale", 1.0), cal.get("offset", [0, 0])
    for bp in spec.get("back_parts") or []:
        if bp.get("at_board"):  # placed from combined evidence (e.g. front + back photos), not the photo alone
            x, y = bp["at_board"]
        else:
            x, y = bp["at_photo"][0] * k + ox, bp["at_photo"][1] * k + oy
        if bp.get("generated") == "screw_head":
            path = _screw_head(os.path.join(config.data_dir(), "boardgen", sid, "screw_head_m2.step"))
            fps.append(f'\t(footprint "objsearch:screw_head_M2" (layer "F.Cu") (at {_f(x)} {_f(y)})\n'
                       + _props(bp["ref"], bp.get("name", "screw")) + _model(path) + "\t)\n")
            sources.append({"part": bp.get("name", bp["ref"]), "model": "generated (gmsh): M2 pan head"})
        else:
            fps.append(_library_footprint(bp["kicad"], (x, y, 0), bp["ref"]))
            sources.append({"part": bp.get("name", bp["ref"]), "footprint": f"KiCad library {bp['kicad']}",
                            "side": "back"})
        flips[bp["ref"]] = bp.get("rot", 0)
    # pin header: plated pads (loose header, as shipped) or a real KiCad header soldered on the back
    hd = spec.get("header")
    if hd:
        (px, py), pitch = hd["pin1"], hd["pitch"]
        if hd.get("state") == "soldered":
            fps.append(_library_footprint(hd["kicad_soldered"], (px, py, 0), "J1H"))
            flips["J1H"] = 0
            sources.append({"part": "pin header (soldered)", "footprint": f"KiCad library {hd['kicad_soldered']}"})
        else:
            pads = "".join(f'\t\t(pad "{i + 1}" thru_hole {"rect" if i == 0 else "oval"} (at 0 {_f(i * pitch)}) '
                           f'(size 1.7 1.7) (drill 1.0) (layers "*.Cu" "*.Mask"))\n' for i in range(hd["pins"]))
            fps.append(f'\t(footprint "objsearch:header_pads_1x{hd["pins"]}" (layer "F.Cu") (at {_f(px)} {_f(py)})\n'
                       + _props("J1H", "DNP") + pads + "\t)\n")
    bs = spec.get("back_silk") or {}
    eff = lambda size: (f'(effects (font (size {_f(size)} {_f(size)}) (thickness {_f(size * 0.15)})) '  # noqa: E731
                        '(justify mirror))')
    for t in bs.get("texts", []):
        x, y, ang = (list(t["at"]) + [0])[:3]
        gr_back = f'\t(gr_text "{t["text"]}" (at {_f(x)} {_f(y)} {_f(ang)}) (layer "B.SilkS") {eff(t.get("size", 1.0))})\n'
        fps.append(gr_back)
    pl = bs.get("pin_labels")
    if pl and hd:
        for i, lab in enumerate(pl["labels"]):
            fps.append(f'\t(gr_text "{lab}" (at {_f(pl["x"])} {_f(hd["pin1"][1] + i * hd["pitch"])} 0) (layer "B.SilkS") '
                       f'{eff(pl.get("size", 1.2))})\n')
    for (x0, y0), (x1, y1) in bs.get("rects", []):
        fps.append(f'\t(gr_rect (start {_f(x0)} {_f(y0)}) (end {_f(x1)} {_f(y1)}) {_stroke(0.15)} (fill no) '
                   '(layer "B.SilkS"))\n')
    # reserved / unpopulated pads: value DNP -> attach_models leaves them as bare copper
    for g in spec.get("dnp_pads") or []:
        pads = "".join(f'\t\t(pad "{i + 1}" thru_hole {shape} (at {_f(x)} {_f(y)}) (size {_f(g["size"])} {_f(g["size"])}) '
                       f'(drill {_f(g["drill"])}) (layers "*.Cu" "*.Mask"))\n' for i, (x, y, shape) in enumerate(g["pads"]))
        silk = ""
        if g.get("ring"):
            cx, cy, r = g["ring"]
            silk += f'\t\t(fp_circle (center {_f(cx)} {_f(cy)}) (end {_f(cx + r)} {_f(cy)}) {_stroke(0.15)} (fill no) (layer "F.SilkS"))\n'
        if g.get("plus"):
            px, py = g["plus"]
            silk += (f'\t\t(fp_text user "+" (at {_f(px)} {_f(py)} 0) (layer "F.SilkS") '
                     '(effects (font (size 0.9 0.9) (thickness 0.15))))\n')
        fps.append(f'\t(footprint "objsearch:{g["name"]}_reserved" (layer "F.Cu") (at 0 0)\n'
                   + _props(g["name"], "DNP") + pads + silk + "\t)\n")
    sp = spec.get("smd_pads")
    if sp:
        pads = "" if sp.get("pads") is False else "".join(
            f'\t\t(pad "{i + 1}" smd rect (at {_f(x)} {_f(y)}) (size {_f(sp["size"][0])} {_f(sp["size"][1])}) '
            f'(layers "F.Cu" "F.Mask"))\n' for i, (x, y) in enumerate(sp["at"]))
        labels = "".join(f'\t\t(fp_text user "{t}" (at {_f(x)} {_f(sp["label_y"])} 0) (layer "F.SilkS") '
                         '(effects (font (size 0.9 0.9) (thickness 0.15))))\n'
                         for t, (x, _) in zip(sp.get("labels", []), sp["at"]))
        fps.append('\t(footprint "objsearch:solder_pads" (layer "F.Cu") (at 0 0)\n' + _props("P1", "DNP") + pads + labels + "\t)\n")
    gr = [f'\t(gr_rect (start {_f(-w / 2)} {_f(-h / 2)}) (end {_f(w / 2)} {_f(h / 2)}) {_stroke(0.1)} (fill no) '
          '(layer "Edge.Cuts"))\n']
    for (x0, y0), (x1, y1) in (spec.get("silk") or {}).get("rects", []):
        gr.append(f'\t(gr_rect (start {_f(x0)} {_f(y0)}) (end {_f(x1)} {_f(y1)}) {_stroke(0.15)} (fill no) (layer "F.SilkS"))\n')
    for t in (spec.get("silk") or {}).get("texts", []):
        x, y, a = (list(t["at"]) + [0])[:3]
        s = t.get("size", 1.0)
        gr.append(f'\t(gr_text "{t["text"]}" (at {_f(x)} {_f(y)} {_f(a)}) (layer "F.SilkS") '
                  f'(effects (font (size {_f(s)} {_f(s)}) (thickness {_f(s * 0.15)}))))\n')
    layers = "\n".join(f"\t\t{x}" for x in (
        '(0 "F.Cu" signal)', '(2 "B.Cu" signal)', '(13 "F.Paste" user)', '(15 "B.Paste" user)',
        '(5 "F.SilkS" user "F.Silkscreen")', '(7 "B.SilkS" user "B.Silkscreen")', '(1 "F.Mask" user)',
        '(3 "B.Mask" user)', '(25 "Edge.Cuts" user)', '(31 "F.CrtYd" user "F.Courtyard")',
        '(29 "B.CrtYd" user "B.Courtyard")', '(35 "F.Fab" user)', '(33 "B.Fab" user)'))
    text = (f'(kicad_pcb\n\t(version 20260206)\n\t(generator "objsearch-boardgen")\n\t(generator_version "10.0")\n'
            f'\t(general\n\t\t(thickness {_f(b.get("thickness", 1.6))})\n\t\t(legacy_teardrops no)\n\t)\n'
            f'\t(paper "A4")\n\t(layers\n{layers}\n\t)\n\t(setup\n\t\t(pad_to_mask_clearance 0)\n\t)\n'
            f'\t(net 0 "")\n' + "".join(fps) + "".join(gr) + ")\n")
    text = kicad.set_board_colors(text, MASK.get(b.get("mask", "green"), b.get("mask")),
                                  SILK.get(b.get("silk", "white"), b.get("silk")),
                                  finish=FINISH.get(b.get("finish", "HASL"), b.get("finish")),
                                  thickness=float(b.get("thickness", 1.6)))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    if flips:
        _flip_to_back(out_path, flips)
    return {"path": out_path, "sources": sources, "back_parts": len(flips)}


def load(spec_path):
    with open(spec_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build(conn, spec_path, render=True):
    """Spec -> .kicad_pcb under <data>/boardgen/<id>/ (tracked). Optionally KiCad renders for a quick look."""
    spec = load(spec_path)
    out = os.path.join(config.data_dir(), "boardgen", spec["id"], f"{spec['id']}.kicad_pcb")
    with track.step("boardgen", object_id=spec["id"], inputs={"spec": os.path.basename(spec_path)}) as s:
        res = build_pcb(spec, out)
        s.set(pcb=track.rel(out), sources=res["sources"])
    if render:
        res["renders"] = {v: kicad.render(out, os.path.splitext(out)[0] + f"_{v}.png", v) for v in ("iso", "top")}
    return res


# ---------------------------------------------------------------- real-STEP overlay (parts KiCad can't place)

def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _mat(hexcol, roughness=0.5, metallic=0.0):
    import trimesh
    h = hexcol.lstrip("#")
    rgb = [_srgb_to_linear(int(h[i:i + 2], 16) / 255) for i in (0, 2, 4)]
    return trimesh.visual.material.PBRMaterial(baseColorFactor=rgb + [1.0], roughnessFactor=float(roughness),
                                               metallicFactor=float(metallic), doubleSided=True)


FINISH_METAL_SRGB = {"ENIG": ("#D6B05A", 0.3), "HAL SnPb": ("#CFCFD1", 0.35), "OSP": ("#C7764A", 0.35)}


def _world_meshes(scene):
    """trimesh scene -> [(geometry name, mesh in world coordinates)]."""
    out = []
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        m = scene.geometry[gname].copy()
        m.apply_transform(T)
        out.append((gname, m))
    return out


def merge_overlay(board_glb, spec, out_glb, finish="HAL SnPb"):
    """KiCad board GLB (sRGB-as-linear colours) + the spec's real-STEP overlay parts + reconstructed extras -> one
    GLB with linear colours. The STEP is aligned by its own PCB part (spec overlay.align_part) onto KiCad's
    board_PCB mesh. glTF frame: x = board X, y = up, z = board Y (KiCad and cascadio both map this way)."""
    import cascadio
    import numpy as np
    import trimesh
    board = trimesh.load(board_glb, force="scene")
    meshes = _world_meshes(board)
    pcb_parts = [m for n, m in meshes if n.startswith("board_PCB")]  # KiCad splits the PCB into face meshes
    (bx0, by0, bz0) = np.min([m.bounds[0] for m in pcb_parts], axis=0)
    (bx1, by1, bz1) = np.max([m.bounds[1] for m in pcb_parts], axis=0)
    cx, cz, top = (bx0 + bx1) / 2, (bz0 + bz1) / 2, by1
    out = trimesh.Scene()
    metal, rough = FINISH_METAL_SRGB.get(finish, FINISH_METAL_SRGB["HAL SnPb"])
    for n, m in meshes:  # KiCad colours are sRGB values written as linear: fix, and give pads their metal
        mat = getattr(m.visual, "material", None)
        if n.startswith("board_pad"):
            m.visual = trimesh.visual.TextureVisuals(material=_mat(metal, rough, 1.0))
        elif mat is not None and getattr(mat, "baseColorFactor", None) is not None:
            f = np.asarray(mat.baseColorFactor, dtype=float) / (255.0 if np.asarray(mat.baseColorFactor).max() > 1 else 1)
            mat.baseColorFactor = [_srgb_to_linear(f[0]), _srgb_to_linear(f[1]), _srgb_to_linear(f[2]), f[3]]
        out.add_geometry(m, geom_name=n)
    ov = spec.get("overlay")
    placed = []
    if ov:
        step = _fetch(ov["url"], spec["id"])
        raw = os.path.splitext(step)[0] + "_raw.glb"
        if not os.path.exists(raw):
            cascadio.step_to_glb(step, raw, tol_linear=0.02)
        parts = _world_meshes(trimesh.load(raw, force="scene"))  # cascadio: raw CAD Z-up, metres

        def size_mm(m):
            return sorted(float(x) * 1000 for x in m.extents)

        def fits(m, want):
            return all(abs(g - w) <= max(0.3, 0.02 * w) for g, w in zip(size_mm(m), sorted(want)))
        ref = next(m for _, m in parts if fits(m, ov["align_part"]))
        (sx0, sy0, sz0), (sx1, sy1, sz1) = ref.bounds
        scx, scy, stop = (sx0 + sx1) / 2, (sy0 + sy1) / 2, sz1
        # raw (x, y, z) -> glTF (x, z, -y), aligned: x on centre, CAD top of PCB on KiCad top, y on centre
        T = np.array([[1, 0, 0, cx - scx], [0, 0, 1, top - stop], [0, -1, 0, cz + scy], [0, 0, 0, 1]], dtype=float)
        for k in ov.get("keep", []):
            hits = [m for _, m in parts if fits(m, k["size"])]
            for i, m in enumerate(hits):
                m = m.copy()
                m.apply_transform(T)
                m.visual = trimesh.visual.TextureVisuals(material=_mat(k["hex"], k.get("roughness", 0.5),
                                                                       k.get("metallic", 0.0)))
                out.add_geometry(m, geom_name=f"overlay_{k['label']}_{i}")
            placed.append((k["label"], len(hits)))
    for e in spec.get("extras") or []:
        x0, x1, y0, y1, z0, z1 = e["box"]
        box = trimesh.creation.box(extents=[(x1 - x0) / 1000, (z1 - z0) / 1000, (y1 - y0) / 1000])
        box.apply_translation([cx + (x0 + x1) / 2000, top + (z0 + z1) / 2000, cz + (y0 + y1) / 2000])
        box.visual = trimesh.visual.TextureVisuals(material=_mat(e["hex"], e.get("roughness", 0.5),
                                                                 e.get("metallic", 0.0)))
        out.add_geometry(box, geom_name=f"extra_{e['label']}")
        placed.append((e["label"], 1))
    os.makedirs(os.path.dirname(os.path.abspath(out_glb)), exist_ok=True)
    out.export(out_glb)
    return placed


def build_model(spec, out_dir):
    """Spec -> final GLB (KiCad board + real-STEP overlay + extras) for the model pipeline. -> (glb path, info)."""
    pcb = build_pcb(spec, os.path.join(out_dir, f"{spec['id']}.kicad_pcb"))["path"]
    glb, info = kicad.board_to_step(pcb, os.path.join(out_dir, "kicad_work"), renders=(), fmt="glb",
                                    object_id=spec["id"])
    if not (spec.get("overlay") or spec.get("extras")):
        return glb, info
    final = os.path.join(out_dir, f"{spec['id']}_replica.glb")
    info["overlay"] = merge_overlay(glb, spec, final, finish=info.get("copper_finish") or "HAL SnPb")
    return final, info
