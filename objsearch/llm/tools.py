"""One tool registry shared by every backend. Schemas use the JSON-schema subset that both Gemini
function declarations and Claude tools accept (type/properties/required/description/enum/items).
Handlers are bound per run with build_tools(); images are referenced by the labels given in the
prompt (e.g. "crop_0"), never passed as bytes in tool arguments."""
from .. import db
from .base import Tool

S = lambda d: {"type": "string", "description": d}  # noqa: E731

SCHEMAS = {
    "web_search": ("Search the web. Use specific queries: part numbers, shield text, FCC IDs, "
                   "'<name> datasheet dimensions', '<name> STEP model'.",
                   {"type": "object", "properties": {"query": S("search query")}, "required": ["query"]}),
    "web_fetch": ("Fetch a web page (that appeared in search results) as text.",
                  {"type": "object", "properties": {"url": S("absolute http(s) URL")}, "required": ["url"]}),
    "catalog_lookup": ("Search the local object catalog by name, alias, MPN or manufacturer.",
                       {"type": "object", "properties": {"text": S("free text")}, "required": ["text"]}),
    "brickognize_identify": ("Identify a LEGO part/set/minifig from one of the prompt images (Brickognize). "
                             "Returns candidate part numbers with scores.",
                             {"type": "object", "properties": {"image": S("image label, e.g. crop_0")},
                              "required": ["image"]}),
    "ldraw_get": ("Look up a LEGO part in the local LDraw library by part number (e.g. 3001).",
                  {"type": "object", "properties": {"part": S("LDraw part number")}, "required": ["part"]}),
    "kicad_find": ("Search the local KiCad packages3D / Espressif 3D model index for an electronic part.",
                   {"type": "object", "properties": {"query": S("part or module name")}, "required": ["query"]}),
    "sketchfab_search": ("Search downloadable Creative-Commons models on Sketchfab.",
                         {"type": "object", "properties": {"query": S("search text")}, "required": ["query"]}),
    "sketchfab_download": ("Download a Sketchfab model (GLB) by uid into quarantine.",
                           {"type": "object", "properties": {"uid": S("Sketchfab model uid")}, "required": ["uid"]}),
    "download": ("Download a 3D model file or archive from a URL into quarantine (allowlisted types only).",
                 {"type": "object", "properties": {"url": S("direct file URL"),
                                                   "license": S("license stated on the page, if any")},
                  "required": ["url"]}),
    "convert": ("Convert a downloaded file (STEP/STL/OBJ/GLB/zip...) into a GLB for rendering.",
                {"type": "object", "properties": {"file_id": S("id returned by download")}, "required": ["file_id"]}),
    "render_candidate": ("Render a converted model from several views at metric scale.",
                         {"type": "object", "properties": {"model_id": S("id returned by convert")},
                          "required": ["model_id"]}),
    "compare": ("Score a converted model against the observed object (dims ratio, visual similarity, "
                "silhouette IoU). Higher is better; >=0.7 is a good match.",
                {"type": "object", "properties": {"model_id": S("id returned by convert")}, "required": ["model_id"]}),
}

IDENTIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "name": S("specific product name, e.g. 'ESP-01 (ESP8266) WiFi module'"),
        "category": {"type": "string", "enum": ["module", "board", "sensor", "actuator", "camera", "storage",
                                                "lego", "power", "prototyping", "other"]},
        "manufacturer": S("manufacturer or brand, or 'unknown'"),
        "mpn": S("manufacturer part number, or ''"),
        "aliases": {"type": "array", "items": {"type": "string"}},
        "expected_dims_mm": {"type": "array", "items": {"type": "number"},
                             "description": "datasheet [length, width, height] in mm"},
        "evidence_urls": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "description": "0..1, calibrated: 0.3 = a guess, 0.9 = markings/part number confirmed"},
        "catalog_id": S("id from the catalog list if it is one of those items, else ''"),
        "ldraw_id": S("LEGO part number if it is a LEGO part, else ''"),
        "reasoning": S("one short paragraph: what you saw and what confirmed it"),
    },
    "required": ["name", "category", "confidence", "evidence_urls", "catalog_id"],
}

ACQUIRE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {"type": "array", "items": {"type": "object", "properties": {
            "model_id": S("id returned by convert"),
            "source_url": S("page the model came from"),
            "license": S("license, or 'unknown'"),
            "score": {"type": "number"}},
            "required": ["model_id", "source_url", "license"]}},
        "notes": S("what was tried and why the best candidate was chosen"),
    },
    "required": ["candidates"],
}

IDENTIFY_TOOLS = ("catalog_lookup", "web_search", "web_fetch", "brickognize_identify", "ldraw_get", "kicad_find")
ACQUIRE_TOOLS = ("catalog_lookup", "web_search", "web_fetch", "ldraw_get", "kicad_find", "sketchfab_search",
                 "sketchfab_download", "download", "convert", "render_candidate", "compare")


def build_tools(handlers: dict, names=None):
    """Bind handlers to schemas. Tools without a handler are left out (e.g. no Sketchfab token)."""
    out = []
    for name in names or SCHEMAS:
        if name in handlers:
            desc, params = SCHEMAS[name]
            out.append(Tool(name, desc, params, handlers[name]))
    return out


def catalog_lookup_handler(conn):
    def handler(args):
        hits = db.search(conn, args["text"], limit=8)
        return {"matches": [{k: h[k] for k in ("id", "name", "category", "mpn", "dims_mm", "aliases", "status")}
                            for h in hits]}
    return handler
