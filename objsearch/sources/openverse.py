"""Openverse (api.openverse.org): openly licensed images from Flickr, Wikimedia Commons and others,
with license + attribution. No key needed: anonymous limits are 20 searches/min, 200/day, and the
thumbnail proxy has its own 1000/day bucket. Direct downloads from Flickr/Wikimedia hosts are refused
for scripts, so images come through the Openverse thumbnail proxy."""
import re
import time

import httpx

from .. import track

API = "https://api.openverse.org/v1/images/"
UA = "objsearch/0.1 (local object-catalog tool; personal use) python-httpx"
MAX_BYTES = 8 * 1024 * 1024
SEARCH_LIMIT_PER_DAY = 190  # stay under the anonymous 200/day
_last = [0.0]


def _pace(min_gap=3.2):
    """20/min burst limit -> one search every ~3 s."""
    wait = _last[0] + min_gap - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.monotonic()


def search(query, n=6, client=None, pace=True):
    conn = track.current_run().conn
    track.check_quota(conn, "openverse_search", SEARCH_LIMIT_PER_DAY)
    if pace:
        _pace()
    c = client or httpx.Client(timeout=30, headers={"User-Agent": UA})
    t0 = time.perf_counter()
    r = c.get(API, params={"q": query, "page_size": n, "mature": "false"})
    track.record_http("GET", str(r.url), r.status_code, len(r.content), None, round((time.perf_counter() - t0) * 1e3))
    track.bump_quota(conn, "openverse", "images", "openverse_search")
    if r.status_code == 429:
        raise track.QuotaExceeded(f"openverse rate limited: {r.headers.get('retry-after', '?')}s")
    r.raise_for_status()
    return [x for x in r.json().get("results", []) if not x.get("mature")]


def fetch_image(result, client=None):
    """Image bytes via the Openverse proxy (full size; we downscale locally)."""
    c = client or httpx.Client(timeout=60, headers={"User-Agent": UA})
    for url in (f"{API}{result['id']}/thumb/?full_size=true", f"{API}{result['id']}/thumb/"):
        t0 = time.perf_counter()
        r = c.get(url, follow_redirects=True)
        data = r.content
        track.record_http("GET", url, r.status_code, len(data), None, round((time.perf_counter() - t0) * 1e3))
        if r.status_code != 424:  # 424: proxy couldn't fetch the full-size upstream; the cached thumb often works
            break
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/"):
        raise ValueError(f"not an image: {r.headers.get('content-type')}")
    if len(data) > MAX_BYTES:
        raise ValueError(f"image too large ({len(data)} bytes)")
    return data


def _norm(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def relevance(result, query):
    """Share of query words (punctuation-insensitive) found in the title or tags: 'ESP-01' matches 'esp01'."""
    toks = [t for t in (_norm(w) for w in query.split()) if t]
    hay = _norm(result.get("title")) + " " + " ".join(_norm(t.get("name")) for t in result.get("tags") or [])
    return sum(t in hay for t in toks) / len(toks) if toks else 0.0
