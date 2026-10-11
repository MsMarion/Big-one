"""Brickognize (api.brickognize.com): LEGO part/set/minifig recognition from a photo. No key needed.
Part ids follow BrickLink/LDraw numbering (e.g. 3001), so they map onto catalog ldraw_id."""
import time

import httpx

from .. import track
from .openverse import UA

API = "https://api.brickognize.com/predict/"


def identify(image_bytes, client=None, top=5):
    c = client or httpx.Client(timeout=60, headers={"User-Agent": UA})
    t0 = time.perf_counter()
    r = c.post(API, files={"query_image": ("photo.jpg", image_bytes, "image/jpeg")})
    track.record_http("POST", API, r.status_code, len(r.content), None, round((time.perf_counter() - t0) * 1e3))
    r.raise_for_status()
    d = r.json()
    items = [{"id": i.get("id"), "name": i.get("name"), "type": i.get("type"), "category": i.get("category"),
              "score": round(float(i.get("score", 0)), 3),
              "url": next((s["url"] for s in i.get("external_sites", []) if s.get("url")), None)}
             for i in d.get("items", [])[:top]]
    box = d.get("bounding_box") or {}
    return {"items": items, "detection_score": round(float(box.get("score", 0)), 3)}
