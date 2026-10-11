"""Guarded page fetch for the agent's web_fetch tool: http(s) only, no local/private hosts, size and
time caps, HTML reduced to readable text. Nothing fetched is executed or stored beyond the text."""
import html.parser
import ipaddress
import re
import socket
import time
import urllib.parse

import httpx

from .. import track
from .openverse import UA

MAX_BYTES = 3 * 1024 * 1024
MAX_CHARS = 8000


class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}

    def __init__(self):
        super().__init__()
        self.out, self.depth, self.title, self._in_title = [], 0, "", False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("p", "br", "li", "tr", "h1", "h2", "h3", "h4", "div", "td", "th"):
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.depth:
            self.out.append(data)

    def text(self):
        t = re.sub(r"[ \t\r\f\v]+", " ", "".join(self.out))
        return re.sub(r"\n\s*\n+", "\n", t).strip()


def _public_host(host):
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _check(url, check_host):
    u = urllib.parse.urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("only http(s) URLs")
    if check_host and not _public_host(u.hostname):
        raise ValueError("refusing to fetch a local/private address")


def _get(url, client=None, check_host=True, max_redirects=5, max_bytes=MAX_BYTES, truncate=True):
    """Redirects are followed by hand so every hop gets the public-host check. -> (url, data, ctype, enc)."""
    c = client or httpx.Client(timeout=30, headers={"User-Agent": UA})
    t0 = time.perf_counter()
    for _ in range(max_redirects + 1):
        _check(url, check_host)
        with c.stream("GET", url, follow_redirects=False) as r:
            if r.is_redirect and r.headers.get("location"):
                url = urllib.parse.urljoin(url, r.headers["location"])
                continue
            chunks, n = [], 0
            for chunk in r.iter_bytes():
                n += len(chunk)
                if n > max_bytes:
                    if not truncate:
                        raise ValueError(f"larger than {max_bytes // 2**20} MB")
                    break
                chunks.append(chunk)
            data, ctype, status, enc = b"".join(chunks), r.headers.get("content-type", ""), r.status_code, r.encoding
        break
    else:
        raise ValueError("too many redirects")
    track.record_http("GET", url, status, len(data), None, round((time.perf_counter() - t0) * 1e3))
    if status >= 400:
        raise ValueError(f"HTTP {status}")
    return url, data, ctype, enc


def download(url, max_bytes=40 * 2**20, client=None, check_host=True):
    """Whole file (documents): same guards as fetch_text, no truncation. -> (final url, bytes, content type)."""
    url, data, ctype, _ = _get(url, client, check_host, max_bytes=max_bytes, truncate=False)
    return url, data, ctype


def fetch_text(url, client=None, check_host=True, max_redirects=5):
    url, data, ctype, enc = _get(url, client, check_host, max_redirects)
    if "pdf" in ctype:
        return {"url": url, "title": "", "text": "(PDF document; text extraction not supported yet)", "truncated": False}
    if "html" not in ctype and "text" not in ctype:
        raise ValueError(f"unsupported content type {ctype or '?'}")
    p = _Text()
    p.feed(data.decode(enc or "utf-8", errors="replace"))
    text = p.text()
    return {"url": url, "title": p.title.strip()[:200], "text": text[:MAX_CHARS], "truncated": len(text) > MAX_CHARS}
