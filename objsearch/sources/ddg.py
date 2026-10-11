"""Keyless web search fallback (DuckDuckGo via the `ddgs` package) for when Google-grounded search
isn't available (free-tier Gemini 3.x has no grounding; the 2.5 models are closed to new users)."""
import time

from .. import track


def search(query, n=6):
    from ddgs import DDGS
    t0 = time.perf_counter()
    try:
        rows = DDGS().text(query, max_results=n) or []
    finally:
        track.record_http("GET", f"ddgs:text?q={query[:120]}", None, None, None, round((time.perf_counter() - t0) * 1e3))
    track.bump_quota(track.current_run().conn, "duckduckgo", "text", "web_search_fallback")
    return {"results": [{"title": r.get("title"), "url": r.get("href"), "snippet": (r.get("body") or "")[:300]}
                        for r in rows if r.get("href")],
            "trust": "fallback (DuckDuckGo results, not verified by Google grounding)"}
