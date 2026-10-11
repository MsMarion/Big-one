"""GitHub code search for CAD files (STEP/STP) - returns DIRECT download links plus each repo's license, which is
what the acquire agent needs (web search returns prose, not files). Uses the GitHub CLI's login (`gh auth login`);
code search requires authentication."""
import functools
import json
import os
import shutil
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

from .. import track

CAD_EXT = ("step", "stp")


def _gh(path, params=None):
    """Run `gh api` (no tracking, safe in worker threads) -> (url, ok, stdout|stderr, bytes, ms)."""
    exe = shutil.which("gh")
    if not exe:
        raise RuntimeError("GitHub CLI (gh) not found on PATH")
    args = [exe, "api", "-X", "GET", path] + sum((["-f", f"{k}={v}"] for k, v in (params or {}).items()), [])
    t0 = time.perf_counter()
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", timeout=60)
    ms = round((time.perf_counter() - t0) * 1e3)
    url = f"github:{path}?{urllib.parse.urlencode(params or {})}"[:300]
    return url, p.returncode == 0, p.stdout if p.returncode == 0 else p.stderr.strip()[:200], len(p.stdout), ms


def _record(res):
    url, ok, body, n, ms = res
    track.record_http("GET", url, 200 if ok else 0, n, None, ms)
    return json.loads(body) if ok else None


_CACHE = {}  # (path, params) -> parsed JSON, for this process (searches repeat within one acquire run)


def _rate_limited(msg):
    m = (msg or "").lower()
    return "rate limit" in m or "secondary rate" in m or "http 429" in m


def _wait_for_reset(max_wait=65):
    """Sleep until the code-search quota resets (GitHub allows 10 code searches a minute)."""
    res = _gh("rate_limit")
    try:
        reset = json.loads(res[2])["resources"]["code_search"]["reset"] if res[1] else time.time() + 30
    except (ValueError, KeyError):
        reset = time.time() + 30
    time.sleep(min(max(1.0, reset - time.time() + 1), max_wait))


def _api(path, params=None):
    key = (path, json.dumps(params or {}, sort_keys=True))
    if key in _CACHE:
        return _CACHE[key]
    res = _gh(path, params)
    if not res[1] and _rate_limited(res[2]):  # wait for the quota window once, then retry
        track.record_http("GET", res[0], 429, 0, None, res[4])
        _wait_for_reset()
        res = _gh(path, params)
    out = _record(res)
    if out is None:
        raise RuntimeError(f"GitHub search failed for {path}: {res[2][:160]}")
    _CACHE[key] = out
    return out


def clean_query(q, max_words=6):
    """Product names -> a code-search query GitHub accepts: quotes, brackets and ':' (qualifier syntax) removed,
    at most max_words words."""
    import re
    return " ".join(re.sub(r'["()\[\]{}:;,]', " ", q or "").split()[:max_words])


def available():
    if os.environ.get("OBJSEARCH_NO_GITHUB", "0") not in ("0", "", "false", "no"):  # tests / offline runs
        return False
    return shutil.which("gh") is not None


@functools.lru_cache(maxsize=256)
def _license_raw(full_name):
    return _gh(f"repos/{full_name}")


def repo_license(full_name, res=None):
    data = _record(res or _license_raw(full_name)) or {}
    lic = (data.get("license") or {}).get("spdx_id")
    return lic if lic and lic != "NOASSERTION" else "unknown (no license file in repo)"


BOARD_EXT = ("kicad_pcb", "brd")


def search_cad(query, limit=8, exts=CAD_EXT):
    """Code search for CAD files matching query -> [{name, repo, url (raw, direct), page, license}].
    exts: CAD_EXT (STEP models) or BOARD_EXT (PCB design files)."""
    out, seen = [], set()
    query = clean_query(query)
    if not query:
        return []
    for ext in exts:
        data = _api("search/code", {"q": f"{query} extension:{ext}", "per_page": limit})
        for it in data.get("items", []):
            page = it["html_url"]
            if page in seen:
                continue
            seen.add(page)
            rest = urllib.parse.unquote(page.split("/blob/", 1)[1])  # "<commit>/<path>"; paths can hold spaces
            out.append({"name": it["path"].split("/")[-1], "repo": it["repository"]["full_name"], "page": page,
                        "url": f"https://raw.githubusercontent.com/{it['repository']['full_name']}/"
                               f"{urllib.parse.quote(rest)}"})
        if len(out) >= limit:
            break
    out = out[:limit]
    repos = sorted({r["repo"] for r in out})
    with ThreadPoolExecutor(6) as ex:  # one license lookup per repo, in parallel; recorded from this thread
        raw = dict(zip(repos, ex.map(_license_raw, repos)))
    lic = {n: repo_license(n, raw[n]) for n in repos}
    for r in out:
        r["license"] = lic[r["repo"]]
    return out
