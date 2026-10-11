"""LDraw.org parts library: an exact preview render for every official LEGO part, the part's author and
license (CC BY 4.0, per each .dat header), and the part's real geometry (part_glb) for 3D models.

Library files are fetched one by one from library.ldraw.org/library/official/{parts,p}/ and cached forever
under <data>/ldraw/ (a brick needs ~10-30 small files the first time, then none)."""
import os
import re
import time

import httpx

from .. import track
from .openverse import UA

BASE = "https://library.ldraw.org"


def _get(c, url):
    t0 = time.perf_counter()
    r = c.get(url, follow_redirects=True)
    track.record_http("GET", url, r.status_code, len(r.content), None, round((time.perf_counter() - t0) * 1e3))
    r.raise_for_status()
    return r


def find_part(part, client=None):
    """-> {image_url, landing_url, author, license, title} for an official part, or None."""
    c = client or httpx.Client(timeout=30, headers={"User-Agent": UA})
    page = _get(c, f"{BASE}/parts/list?tableSearch={part}").text
    m = re.search(rf"{re.escape(BASE)}/media/parts/(\d+)/conversions/{re.escape(part)}-thumb\.png", page)
    if not m:
        return None
    dbid = m.group(1)
    head = _get(c, f"{BASE}/library/official/parts/{part}.dat").text[:2000]
    field = lambda pat: (re.search(pat, head, re.M) or [None, None])[1]  # noqa: E731
    title = (field(r"^0 (?!Name:|Author:|!)(.+)$") or part).strip()
    author = (field(r"^0 Author:\s*(.+)$") or "LDraw.org contributors").strip()
    lic = (field(r"^0 !LICENSE (.+)$") or "Licensed under CC BY 4.0").strip()
    ver = (re.search(r"CC BY(?:-SA)? ([\d.]+)", lic) or [None, "4.0"])[1]
    return {"image_url": f"{BASE}/media/parts/{dbid}/{part}.png", "landing_url": f"{BASE}/parts/{dbid}",
            "author": author, "title": f"LDraw {part}: {re.sub(r' +', ' ', title)}", "license": "by",
            "license_version": ver, "license_url": f"https://creativecommons.org/licenses/by/{ver}/", "id": f"ldraw:{part}"}


def fetch_image(info, client=None):
    c = client or httpx.Client(timeout=30, headers={"User-Agent": UA})
    r = _get(c, info["image_url"])
    if not r.headers.get("content-type", "").startswith("image/"):
        raise ValueError(f"not an image: {r.headers.get('content-type')}")
    return r.content


# ---------------------------------------------------------------- geometry

LDU_MM = 0.4  # 1 LDraw unit = 0.4 mm; LDraw is -Y up
LICENSE = "CC BY 4.0 (LDraw.org parts library)"


class _Lib:
    def __init__(self, root, client=None):
        self.root, self.c, self.text = root, client, {}

    def _client(self):
        if self.c is None:
            self.c = httpx.Client(timeout=30, headers={"User-Agent": UA})
        return self.c

    def get(self, ref):
        """'s\\3001s01.dat' / 'stud.dat' / '48\\4-4cyli.dat' -> file text (cache, else parts/ then p/ online)."""
        ref = ref.strip().replace("\\", "/").lower()
        if ref in self.text:
            return self.text[ref]
        for d in ("parts", "p"):
            path = os.path.join(self.root, d, *ref.split("/"))
            if os.path.exists(path):
                with open(path, encoding="utf-8", errors="replace") as f:
                    self.text[ref] = f.read()
                return self.text[ref]
        seen = []
        for d in ("parts", "p"):
            url = f"{BASE}/library/official/{d}/{ref}"
            for attempt in range(4):  # the library rate-limits bursts of small requests
                t0 = time.perf_counter()
                r = self._client().get(url, follow_redirects=True)
                track.record_http("GET", url, r.status_code, len(r.content), None,
                                  round((time.perf_counter() - t0) * 1e3))
                if r.status_code not in (429, 500, 502, 503, 504):
                    break
                time.sleep(float(r.headers.get("retry-after") or 0) or 1.5 * (attempt + 1))
            seen.append(f"{d}:{r.status_code}")
            if r.status_code == 200 and r.text.lstrip().startswith("0"):
                path = os.path.join(self.root, d, *ref.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                self.text[ref] = r.text
                return r.text
        raise FileNotFoundError(f"LDraw file {ref} not in the official library ({', '.join(seen)})")

    def colors(self):
        """LDConfig: code -> (r, g, b, a)."""
        if "_colors" not in self.text:
            path = os.path.join(self.root, "LDConfig.ldr")
            if not os.path.exists(path):
                r = self._client().get(f"{BASE}/library/official/LDConfig.ldr", follow_redirects=True)
                track.record_http("GET", str(r.url), r.status_code, len(r.content), None, None)
                r.raise_for_status()
                os.makedirs(self.root, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)
            cols = {}
            with open(path, encoding="utf-8") as f:
                for line in f:
                    m = re.match(r"0 !COLOUR (\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})(?:.*ALPHA\s+(\d+))?", line)
                    if m:
                        h = m.group(3)
                        cols[int(m.group(2))] = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16),
                                                 int(m.group(4) or 255), m.group(1).lower())
            self.text["_colors"] = cols
        return self.text["_colors"]


def color_code(names, lib):
    """Catalog color names (e.g. ['red']) -> LDraw color code; red (4) when nothing matches."""
    cols = lib.colors()
    for n in names or []:
        n = str(n).lower().replace(" ", "_")
        for code, c in cols.items():
            if c[4] == n:
                return code
        for code, c in sorted(cols.items()):
            if n in c[4].split("_"):
                return code
    return 4


def triangles(part, lib, color=4):
    """Flatten a part into triangles: -> (tris float32 [N,3,3] in LDU, LDraw coords; codes int [N])."""
    import numpy as np
    tris, codes = [], []

    def walk(ref, m, col, invert, depth):
        if depth > 32:
            raise ValueError("LDraw nesting too deep")
        ccw, invert_next = True, False
        for line in lib.get(ref).splitlines():
            t = line.split()
            if not t:
                continue
            if t[0] == "0":
                if len(t) >= 3 and t[1] == "BFC":
                    words = t[2:]
                    if "INVERTNEXT" in words:
                        invert_next = True
                    if "CW" in words:
                        ccw = False
                    if "CCW" in words:
                        ccw = True
                continue
            c = int(t[1], 0) if t[0] in "1234" and len(t) > 1 else col
            c = col if c == 16 else c
            if t[0] == "1" and len(t) >= 15:
                x, y, z, a, b, cc, d, e, f, g, h, i = (float(v) for v in t[2:14])
                sub = np.array([[a, b, cc, x], [d, e, f, y], [g, h, i, z], [0, 0, 0, 1]])
                walk(" ".join(t[14:]), m @ sub, c, invert ^ invert_next, depth + 1)
                invert_next = False
            elif t[0] in ("3", "4"):
                n = int(t[0])
                pts = np.array([float(v) for v in t[2:2 + 3 * n]]).reshape(n, 3)
                pts = (m[:3, :3] @ pts.T).T + m[:3, 3]
                flip = (not ccw) ^ invert ^ (np.linalg.det(m[:3, :3]) < 0)
                for tri in ((0, 1, 2), (0, 2, 3))[: n - 2]:
                    p = pts[list(tri)]
                    tris.append(p[::-1] if flip else p)
                    codes.append(c)
            # 2/5: edge and conditional lines: not geometry

    walk(part if part.endswith(".dat") else part + ".dat", np.eye(4), color, False, 0)
    return np.array(tris, dtype=np.float32), np.array(codes)


def part_glb(part, out_path, color_names=None, root=None, client=None):
    """Official LDraw part -> GLB in millimetres, Y-up (glTF), one material per LDraw color. -> info dict."""
    import numpy as np
    import trimesh
    from .. import config
    lib = _Lib(root or os.path.join(config.data_dir(), "ldraw"), client)
    code = color_code(color_names, lib)
    tris, codes = triangles(part, lib, code)
    if not len(tris):
        raise ValueError(f"LDraw part {part} has no faces")
    # LDraw (-Y up) -> glTF (Y up): rotate 180 deg about X (a proper rotation, so winding is kept); LDU -> mm
    v = tris.reshape(-1, 3) * np.array([1, -1, -1]) * LDU_MM
    cols = lib.colors()
    scene = trimesh.Scene()
    for c in sorted(set(codes.tolist())):
        sel = np.repeat(codes == c, 3)
        verts = v[sel]
        r, g, b, a = cols.get(c, cols.get(code))[:4]
        lin = [((x / 255) / 12.92 if x / 255 <= 0.04045 else ((x / 255 + 0.055) / 1.055) ** 2.4) for x in (r, g, b)]
        mat = trimesh.visual.material.PBRMaterial(baseColorFactor=lin + [a / 255], metallicFactor=0.0,  # glTF: linear
                                                  roughnessFactor=0.35, doubleSided=True)
        # vertices stay per-triangle (flat shading): merging would average normals across the brick's hard edges
        mesh = trimesh.Trimesh(verts, np.arange(len(verts)).reshape(-1, 3), process=False)
        mesh.visual = trimesh.visual.TextureVisuals(material=mat)
        scene.add_geometry(mesh, geom_name=f"ldraw_{c}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    scene.export(out_path)
    ext = v.max(axis=0) - v.min(axis=0)
    return {"path": out_path, "triangles": len(tris), "color": code, "color_name": cols.get(code, (0,) * 5)[4],
            "size_mm": [round(float(x), 2) for x in ext]}
