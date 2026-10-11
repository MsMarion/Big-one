"""Public reference images for catalog objects, so you can recognise what you own at a glance.
Each image keeps its source page, creator, license and an attribution line (shown in the dashboard
and gallery). The first image becomes the chosen one; pick another or drop wrong ones on the
object page.

    python -m objsearch refimages                  # objects without reference images
    python -m objsearch refimages --ids esp-01 lego-3001 --force --query "lego 2x4 brick"
"""
import io
import logging
import os

import httpx
from PIL import Image, ImageOps

from . import config, db, photos, track
from .sources import ldraw, openverse

log = logging.getLogger("objsearch.refimages")


def default_query(o):
    return o.get("ref_query") or o.get("mpn") or o["name"]


def _fallbacks(q):
    """Shorter queries when the full one finds nothing: drop words from the end."""
    words = q.split()
    return [" ".join(words[:i]) for i in range(len(words), 0, -1) if len(" ".join(words[:i])) >= 3]


def _store(object_id, idx, data):
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(data)))
    if im.mode in ("RGBA", "LA", "P"):  # transparent renders (LDraw) -> white background
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    im = im.convert("RGB")
    rel = f"refs/{object_id}/{idx}.jpg"
    full = os.path.join(config.data_dir(), rel)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    big = im.copy()
    big.thumbnail((900, 900))
    big.save(full, "JPEG", quality=88)
    im.thumbnail((360, 360))
    im.save(os.path.join(config.data_dir(), photos.thumb_of(rel)), "JPEG", quality=85)
    return rel


def fetch_for_object(conn, oid, n=3, query=None, force=False, client=None, pace=True, max_searches=2,
                     min_relevance=0.66):
    """Search + download up to n images for one object. Returns number stored."""
    o = db.get_object(conn, oid)
    if o is None:
        raise KeyError(oid)
    existing = conn.execute("SELECT COUNT(*) FROM ref_images WHERE object_id=?", (oid,)).fetchone()[0]
    if o.get("ldraw_id") and not query:
        return _fetch_ldraw(conn, o, force or not existing, client)
    with track.step("ref_images", object_id=oid, inputs={"query": query or default_query(o), "n": n}) as s:
        if existing and not force and not query:
            s.skip("already has reference images")
        # rank by word overlap; broaden the query (max 2 searches) until there are enough good matches
        results, seen, used = [], set(), None
        for q in ([query] if query else _fallbacks(default_query(o)))[:max_searches]:
            for res in openverse.search(q, n=20, client=client, pace=pace):
                score = openverse.relevance(res, q)
                if res["id"] not in seen and score >= min_relevance:
                    seen.add(res["id"])
                    results.append((score, len(results), res))
            used = used or (q if results else None)
            if len(results) >= n + 2:
                break
        results = [r for _, _, r in sorted(results, key=lambda x: (-x[0], x[1]))]
        if not results:
            s.skip("no openly licensed images matched")
        have_ids = {r[0] for r in conn.execute("SELECT provider_id FROM ref_images WHERE object_id=?", (oid,))}
        rank0 = conn.execute("SELECT COALESCE(MAX(rank),0) FROM ref_images WHERE object_id=?", (oid,)).fetchone()[0]
        stored, errors = 0, []
        for res in results:
            if stored >= n:
                break
            if res["id"] in have_ids:
                continue
            try:
                data = openverse.fetch_image(res, client=client)
                rel = _store(oid, f"{res['id'][:8]}", data)
            except (httpx.HTTPError, ValueError, OSError) as e:
                errors.append(f"{res['id'][:8]}: {type(e).__name__}: {e}")
                continue
            stored += 1
            db.add_ref_image(conn, oid, path=rel, title=res.get("title"),
                             provider=f"openverse/{res.get('source') or res.get('provider')}", provider_id=res["id"],
                             landing_url=res.get("foreign_landing_url"), image_url=res.get("url"),
                             creator=res.get("creator"), creator_url=res.get("creator_url"),
                             license=res.get("license"), license_version=res.get("license_version"),
                             license_url=res.get("license_url"), attribution=res.get("attribution"),
                             query=used, rank=rank0 + stored, chosen=0)
        if stored and not conn.execute("SELECT 1 FROM ref_images WHERE object_id=? AND chosen=1", (oid,)).fetchone():
            first = conn.execute("SELECT id FROM ref_images WHERE object_id=? ORDER BY rank LIMIT 1", (oid,)).fetchone()
            db.choose_ref_image(conn, oid, first[0], actor="pipeline")
        s.set(query=used, found=len(results), stored=stored, errors=errors)
        return stored
    return 0  # skipped


def _fetch_ldraw(conn, o, run, client):
    """Exact LDraw part render as the chosen reference (LEGO parts photograph too generically)."""
    oid, part = o["id"], o["ldraw_id"]
    with track.step("ref_images", object_id=oid, inputs={"source": "ldraw", "part": part}) as s:
        if not run or conn.execute("SELECT 1 FROM ref_images WHERE object_id=? AND provider_id=?",
                                   (oid, f"ldraw:{part}")).fetchone():
            s.skip("LDraw render already stored")
        info = ldraw.find_part(part, client)
        if info is None:
            s.skip(f"part {part} not found in the official LDraw library")
        rel = _store(oid, f"ldraw-{part}", ldraw.fetch_image(info, client))
        attribution = (f'"{info["title"]}" by {info["author"]}, LDraw.org Parts Library, licensed under '
                       f'CC BY {info["license_version"]}.')
        rank0 = conn.execute("SELECT COALESCE(MIN(rank),1) FROM ref_images WHERE object_id=?", (oid,)).fetchone()[0]
        rid = db.add_ref_image(conn, oid, path=rel, title=info["title"], provider="ldraw.org", provider_id=info["id"],
                               landing_url=info["landing_url"], image_url=info["image_url"], creator=info["author"],
                               creator_url="https://www.ldraw.org/", license=info["license"],
                               license_version=info["license_version"], license_url=info["license_url"],
                               attribution=attribution, query=f"ldraw {part}", rank=rank0 - 1, chosen=0)
        db.choose_ref_image(conn, oid, rid, actor="pipeline")
        s.set(source="ldraw", part=part, stored=1)
        return 1
    return 0


def fetch_all(conn, ids=None, n=3, force=False, query=None):
    """One tracked run over many objects; a failure on one object doesn't stop the rest."""
    targets = ids or [o["id"] for o in db.list_objects(conn, flag=None if force else "no_ref")]
    totals = {"objects": len(targets), "stored": 0, "skipped": 0, "failed": 0}
    with track.run(conn, "refimages", provider="openverse", notes=f"reference images for {len(targets)} objects") as r:
        client = httpx.Client(timeout=60, headers={"User-Agent": openverse.UA})
        for oid in targets:
            try:
                got = fetch_for_object(conn, oid, n=n, query=query, force=force, client=client)
                if got:
                    totals["stored"] += got
                else:
                    totals["skipped"] += 1
            except track.QuotaExceeded as e:
                log.warning(f"stopping: {e}")
                break
            except Exception as e:  # keep going; the failed step is in the run
                totals["failed"] += 1
                log.warning(f"{oid}: {type(e).__name__}: {e}")
        totals["run_id"] = r.id
    return totals
