"""Physical facts every object needs for a model and a simulation, with where each value came from.

Size ladder (ensure_dims) - an object never lacks dims_mm, so a model can always be built:
    catalog (seed / user) > datasheet > measured (RGB-D sightings) > photo_estimate (vision LLM) > category_default
Physics (physics_for) - mass, friction, restitution and materials for the sim-ready USD layer:
    researched values (objsearch.docs: datasheet / vendor page) override per-category estimates.
"""
import json
import logging
import os
import statistics

from . import config, db, track

log = logging.getLogger("objsearch.physical")

# typical size [L, W, H] mm, effective density g/cm^3 of the bounding box (bbox volume x fill x density = mass),
# surface friction / restitution and the dominant materials. Rough on purpose: "rigged" sim defaults.
CATEGORY = {
    "module":      dict(dims=[25, 18, 3.5], density=1.9, fill=0.45, friction=0.5, restitution=0.1,
                        materials=["FR4 PCB", "metal shield (tinplate)", "solder"]),
    "board":       dict(dims=[68, 53, 14], density=1.9, fill=0.25, friction=0.5, restitution=0.1,
                        materials=["FR4 PCB", "plastic connectors (PBT)", "metal (USB/jacks)"]),
    "sensor":      dict(dims=[30, 20, 12], density=1.6, fill=0.35, friction=0.5, restitution=0.1,
                        materials=["FR4 PCB", "plastic housing (ABS)", "pin headers (brass, PBT)"]),
    "actuator":    dict(dims=[32, 12, 30], density=1.4, fill=0.6, friction=0.6, restitution=0.1,
                        materials=["plastic housing (ABS/nylon)", "steel/brass gears", "copper windings"]),
    "camera":      dict(dims=[70, 30, 30], density=0.9, fill=0.5, friction=0.6, restitution=0.15,
                        materials=["plastic housing (ABS/PC)", "glass lens", "PCB"]),
    "storage":     dict(dims=[40, 14, 7], density=1.3, fill=0.8, friction=0.5, restitution=0.15,
                        materials=["plastic housing (ABS/PC)", "metal USB plug (steel)", "PCB + flash"]),
    "power":       dict(dims=[48, 26, 17], density=1.6, fill=0.85, friction=0.5, restitution=0.1,
                        materials=["plastic/steel casing", "battery chemistry"]),
    "prototyping": dict(dims=[83, 55, 9], density=1.2, fill=0.6, friction=0.7, restitution=0.1,
                        materials=["ABS plastic", "phosphor-bronze contacts"]),
    "display":     dict(dims=[52, 36, 4], density=2.0, fill=0.6, friction=0.5, restitution=0.1,
                        materials=["glass panel", "FR4 PCB", "steel/plastic frame", "polyimide FPC"]),
    "lego":        dict(dims=[16, 16, 11.4], density=1.05, fill=0.45, friction=0.4, restitution=0.3,
                        materials=["ABS plastic"]),
}
DEFAULT = dict(dims=[50, 30, 20], density=1.2, fill=0.5, friction=0.5, restitution=0.1, materials=["unknown"])
DIMS_SOURCES = ("catalog", "datasheet", "model_envelope", "measured", "photo_estimate", "category_default")


def category(o):
    return CATEGORY.get((o.get("category") or "").lower(), DEFAULT)


def best_photo(conn, oid):
    """Absolute path of the best picture of the object: the user's own photo, then a photo from an identify run
    (the actual item, even if not confirmed yet), then the chosen reference image."""
    d = config.data_dir()
    for sql in ("SELECT path FROM photos WHERE object_id=? AND source='user' ORDER BY (kind='ref') DESC, id LIMIT 1",
                "SELECT path FROM photos WHERE object_id=? AND source='run' ORDER BY id DESC LIMIT 1",
                "SELECT path FROM ref_images WHERE object_id=? ORDER BY chosen DESC, rank LIMIT 1"):
        try:
            r = conn.execute(sql, (oid,)).fetchone()
        except Exception:  # older DB without the table
            r = None
        if r and r[0] and os.path.exists(os.path.join(d, r[0])):
            return os.path.join(d, r[0])
    return None


def _measured(conn, oid):
    vals = [json.loads(r[0]) for r in conn.execute(
        "SELECT observed_dims_mm FROM sightings WHERE object_id=? AND observed_dims_mm IS NOT NULL "
        "AND status != 'rejected'", (oid,))]
    vals = [sorted(v, reverse=True) for v in vals if v and len(v) == 3]
    if not vals:
        return None
    return [round(statistics.median(v[i] for v in vals), 1) for i in range(3)]


DIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "dims_mm": {"type": "array", "items": {"type": "number"}, "description": "[L, W, H] in mm, as it lies"},
        "basis": {"type": "string", "description": "what the estimate is based on (known product size, scale cues)"},
    },
    "required": ["dims_mm", "basis"],
}


def _photo_estimate(conn, o, backend):
    from .llm.base import Budget, Image
    photo = best_photo(conn, o["id"])
    content = [f"Product: {o['name']} (category {o.get('category') or '?'}). Estimate its physical size in mm as "
               "[length, width, height] lying flat on a table. If you know the product's real size, use it; "
               "otherwise judge from scale cues in the photo (pin headers 2.54 mm pitch, USB plugs 12 mm wide, "
               "LEGO studs 8 mm apart)."]
    if photo:
        mime = "image/png" if photo.lower().endswith(".png") else "image/jpeg"
        content.append(Image(path=photo, mime=mime, label="photo"))
    res = backend.run("You estimate physical sizes of small objects. Answer with submit_result.", content, [],
                      DIMS_SCHEMA, Budget(max_llm_calls=2, max_tool_calls=0, max_searches=0))
    d = (res.output or {}).get("dims_mm")
    if d and len(d) == 3 and all(isinstance(x, (int, float)) and 0.5 <= x <= 2000 for x in d):
        return [round(float(x), 1) for x in d], res.output.get("basis")
    return None, None


def ensure_dims(conn, oid, backend=None):
    """Make sure the object has dims_mm. Returns (dims_mm, dims_source); records how it was obtained."""
    o = db.get_object(conn, oid)
    if o.get("dims_mm"):
        if not o.get("dims_source"):
            db.update_object(conn, oid, {"dims_source": "catalog"}, actor="pipeline")
        return o["dims_mm"], o.get("dims_source") or "catalog"
    with track.step("dims", object_id=oid) as s:
        dims, src, note = _measured(conn, oid), "measured", "median of RGB-D sightings"
        if not dims and backend is not None:
            try:
                dims, note = _photo_estimate(conn, o, backend)
                src = "photo_estimate"
            except Exception as e:  # no key / quota: fall through to the category default
                log.info(f"{oid}: photo size estimate failed: {e}")
                dims = None
        if not dims:
            dims, src, note = list(category(o)["dims"]), "category_default", f"typical {o.get('category') or 'object'}"
        s.set(dims_mm=dims, dims_source=src, basis=note)
        db.update_object(conn, oid, {"dims_mm": dims, "dims_source": src}, actor="pipeline",
                         reason=f"size from {src}: {note}"[:200])
    return dims, src


def physics_for(o):
    """Physics for the USD layer: researched values (o['physics']) over category estimates. Mass from the
    bounding box when not researched."""
    cat = category(o)
    have = o.get("physics") or {}
    dims = o.get("dims_mm") or cat["dims"]
    est_mass = dims[0] * dims[1] * dims[2] / 1000 * cat["fill"] * cat["density"]  # g
    out = {
        "mass_g": have.get("mass_g") or round(est_mass, 2),
        "mass_source": have.get("mass_source") or "estimate (bounding box x category density)",
        "static_friction": have.get("static_friction") or cat["friction"],
        "dynamic_friction": have.get("dynamic_friction") or round(0.8 * (have.get("static_friction") or cat["friction"]), 3),
        "restitution": have.get("restitution") if have.get("restitution") is not None else cat["restitution"],
        "materials": have.get("materials") or [{"part": "body", "material": m} for m in cat["materials"]],
        "source": have.get("source") or "category estimate",
    }
    return out


BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "box_2d": {"type": "array", "items": {"type": "number"},
                   "description": "tight box around the object: [ymin, xmin, ymax, xmax], each 0-1000"},
    },
    "required": ["found", "box_2d"],
}


def crop_to_object(photo, name, backend, out_path, pad=0.04):
    """One vision call: where is the object in the photo? -> path of a tight crop (or None). Used so a
    placeholder box shows the item on top, not the whole desk."""
    from PIL import Image as PILImage, ImageOps
    from .llm.base import Budget, Image
    mime = "image/png" if photo.lower().endswith(".png") else "image/jpeg"
    res = backend.run("You locate objects in photos. Answer with submit_result.",
                      [f"Find the {name} in this photo and give its tight bounding box.",
                       Image(path=photo, mime=mime, label="photo")], [], BOX_SCHEMA,
                      Budget(max_llm_calls=2, max_tool_calls=0, max_searches=0))
    out = res.output or {}
    b = out.get("box_2d") or []
    if not out.get("found") or len(b) != 4:
        return None
    y0, x0, y1, x1 = (min(max(float(v), 0), 1000) / 1000 for v in b)
    if y1 - y0 < 0.02 or x1 - x0 < 0.02:
        return None
    im = ImageOps.exif_transpose(PILImage.open(photo)).convert("RGB")
    w, h = im.size
    box = (max(0, (x0 - pad) * w), max(0, (y0 - pad) * h), min(w, (x1 + pad) * w), min(h, (y1 + pad) * h))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.crop(tuple(int(v) for v in box)).save(out_path, quality=92)
    return out_path
