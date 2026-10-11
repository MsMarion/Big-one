"""Offline fakes for tests and the dashboard demo: canned search/fetch/model-source tool handlers
and placeholder crop images. Nothing here touches the network."""
import hashlib
import os
import time

from PIL import Image, ImageDraw

from . import track
from .llm import tools

SEARCH = {
    "webcam": [{"title": "1080p USB webcam with clip — product page", "url": "https://example.test/webcam-1080p",
                "snippet": "Black USB 2.0 webcam, 72 x 31 x 30 mm body, universal monitor clip."}],
    "esp": [{"title": "ESP-01 ESP8266 module datasheet", "url": "https://example.test/esp-01.pdf",
             "snippet": "24.8 x 14.3 mm, 2x4 header, PCB antenna."}],
}
MODELS = {"webcam": [{"uid": "fakeuid01", "name": "USB Webcam (low poly)", "license": "CC-BY-4.0",
                      "url": "https://example.test/models/fakeuid01"}]}


def _hit(table, text):
    text = text.lower()
    for k, v in table.items():
        if k in text:
            return v
    return []


def fake_handlers(conn, fail_tool=None, raise_in_step=None):
    """Tool handlers backed by canned data. fail_tool: that tool raises. raise_in_step: the nested
    tracked step of that tool name (download/convert/compare) fails after starting."""
    files, models = {}, {}

    def maybe_fail(name):
        if name == fail_tool:
            raise ConnectionError(f"fake {name} failure")

    def web_search(a):
        maybe_fail("web_search")
        track.bump_quota(track.current_run().conn, "gemini", "fake-search", "grounded_search")
        return {"results": _hit(SEARCH, a["query"]), "trust": "fake"}

    def web_fetch(a):
        maybe_fail("web_fetch")
        track.record_http("GET", a["url"], 200, 2048, None, 12)
        return {"url": a["url"], "text": "Fake page. Dimensions: 72 x 31 x 30 mm. Interface: USB 2.0."}

    def sketchfab_search(a):
        maybe_fail("sketchfab_search")
        return {"results": _hit(MODELS, a["query"])}

    def download(a):
        maybe_fail("download")
        with track.step("download", inputs={"url": a["url"]}) as s:
            data = f"fake model from {a['url']}".encode()
            sha = hashlib.sha256(data).hexdigest()
            track.record_http("GET", a["url"], 200, len(data), sha, 35)
            if raise_in_step == "download":
                raise IOError("connection reset while downloading")
            fid = f"file{len(files) + 1}"
            files[fid] = {"url": a["url"], "sha256": sha, "license": a.get("license", "unknown")}
            s.set(file_id=fid, sha256=sha, bytes=len(data))
        return {"file_id": fid, "sha256": sha, "bytes": len(data), "ext": ".glb"}

    def sketchfab_download(a):
        return download({"url": f"https://example.test/models/{a['uid']}.glb", "license": "CC-BY-4.0"})

    def convert(a):
        maybe_fail("convert")
        with track.step("convert", inputs=a) as s:
            if a["file_id"] not in files:
                raise KeyError(f"unknown file_id {a['file_id']}")
            if raise_in_step == "convert":
                raise ValueError("unsupported STEP schema AP242 in fake converter")
            time.sleep(0.02)
            mid = f"model{len(models) + 1}"
            models[mid] = {**files[a["file_id"]], "tris": 12450, "dims_mm": [71.5, 30.8, 29.6]}
            s.set(model_id=mid, tris=12450)
        return {"model_id": mid, "tris": 12450, "dims_mm": [71.5, 30.8, 29.6]}

    def compare(a):
        maybe_fail("compare")
        with track.step("compare", inputs=a) as s:
            if a["model_id"] not in models:
                raise KeyError(f"unknown model_id {a['model_id']}")
            sc = {"dims_ratio": 0.98, "visual_sim": 0.71, "silhouette_iou": 0.77, "score": 0.81}
            s.set(**sc)
        return sc

    return {"web_search": web_search, "web_fetch": web_fetch, "catalog_lookup": tools.catalog_lookup_handler(conn),
            "sketchfab_search": sketchfab_search, "sketchfab_download": sketchfab_download, "download": download,
            "convert": convert, "compare": compare}


def make_crop(path, label, color, size=(256, 192)):
    """Placeholder 'camera crop': a colored object on a table-grey background with a caption."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im = Image.new("RGB", size, (200, 198, 192))
    d = ImageDraw.Draw(im)
    w, h = size
    d.rounded_rectangle([w * .18, h * .22, w * .82, h * .72], radius=10, fill=color, outline=(40, 40, 40), width=2)
    d.text((10, h - 22), label, fill=(30, 30, 30))
    im.save(path, quality=88)
    return path
