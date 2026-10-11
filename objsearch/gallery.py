"""Static, JS-free gallery of every object (photo + model thumbnail, dims, ids, status, flags).
Images are embedded as small data URIs so the single HTML file can be shared as is.

    python -m objsearch gallery [--out objsearch_data/gallery.html] [--filter owned]
"""
import base64
import html
import io
import os

from PIL import Image

from . import config, db, grasp, photos

CSS = """
:root{color-scheme:light;--s0:#f4f3f0;--s1:#fcfcfb;--b:#e2e1dc;--t:#0b0b0b;--t2:#52514e;--tm:#8a8984;
--acc:#2a78d6;--good:#0ca30c;--warn:#b07800;--crit:#d03b3b}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--s0:#111110;--s1:#1a1a19;--b:#2e2e2b;--t:#fff;
--t2:#c3c2b7;--tm:#8d8c84;--acc:#3987e5;--warn:#fab219}}
*{box-sizing:border-box}body{margin:0;background:var(--s0);color:var(--t);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--b);background:var(--s1)}h1{font-size:18px;margin:0}
main{max-width:1400px;margin:0 auto;padding:16px 20px 40px}h2{font-size:15px;margin:24px 0 10px;color:var(--t2)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}
.card{background:var(--s1);border:1px solid var(--b);border-radius:10px;overflow:hidden}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:var(--b)}
.imgs div{aspect-ratio:4/3;background:var(--s0);display:flex;align-items:center;justify-content:center;
color:var(--tm);font-size:12px}.imgs img{width:100%;height:100%;object-fit:cover;display:block}
.body{padding:10px 12px}.name{font-weight:600}.meta{color:var(--t2);font-size:12px;margin-top:4px}
.tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}.tag{font-size:11px;border:1px solid var(--b);
border-radius:999px;padding:1px 8px;color:var(--t2)}.tag.flag{border-color:var(--warn);color:var(--warn)}
.tag.ok{border-color:var(--good);color:var(--good)}.tag.no{border-color:var(--crit);color:var(--crit)}.muted{color:var(--tm)}
.credit{font-size:10.5px;color:var(--tm);margin-top:6px;overflow-wrap:anywhere}
@media (max-width:600px){main,header{padding-left:16px;padding-right:16px}}
"""


def _data_uri(rel, side=320):
    if not rel:
        return None
    full = os.path.join(config.data_dir(), rel)
    thumb = os.path.join(config.data_dir(), photos.thumb_of(rel))
    src = thumb if os.path.exists(thumb) else full
    if not os.path.exists(src):
        return None
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except OSError:
        return None


def _img(rel, empty):
    uri = _data_uri(rel)
    return f'<div><img src="{uri}" alt=""></div>' if uri else f"<div>{empty}</div>"


def _card(o):
    e = html.escape
    dims = "×".join(f"{d:g}" for d in o["dims_mm"]) + " mm" if o.get("dims_mm") else "dims unknown"
    ids = " · ".join(e(str(o[k])) for k in ("mpn", "ldraw_id", "fcc_id") if o.get(k))
    tags = [f'<span class="tag">{e(o["origin"])}</span>',
            f'<span class="tag{" ok" if o["status"] == "verified" else ""}">{e(o["status"])}</span>']
    g = o.get("grasp") or {}
    if g.get("verdict"):
        cls = {"good": " ok", "borderline": " flag", "no": " no"}[g["verdict"]]
        label = {"good": "gripper: good", "borderline": "gripper: borderline", "no": "gripper: can't pick"}[g["verdict"]]
        tags.append(f'<span class="tag{cls}" title="{e("; ".join(g["reasons"]))}">{label}</span>')
    if o["owned"]:
        tags.append('<span class="tag ok">owned</span>')
    if not o["photo_count"]:
        tags.append('<span class="tag flag">no photo</span>')
    if not o["asset_count"]:
        tags.append('<span class="tag flag">no model</span>')
    if not o.get("dims_mm"):
        tags.append('<span class="tag flag">dims unconfirmed</span>')
    if o["sighting_count"]:
        tags.append(f'<span class="tag">seen {o["sighting_count"]}×</span>')
    ref = not o.get("photo_path") and o.get("ref_path")
    credit = f'<div class="credit">reference image: {e(o.get("ref_attribution") or "")}</div>' if ref else ""
    return (f'<div class="card"><div class="imgs">{_img(o.get("photo_path") or o.get("ref_path"), "no photo")}'
            f'{_img(o.get("thumb_path"), "no model")}</div><div class="body"><div class="name">{e(o["name"])}</div>'
            f'<div class="meta">{e(dims)}{" · " + ids if ids else ""}</div>'
            f'<div class="tags">{"".join(tags)}</div>{credit}</div></div>')


def render(conn, flag=None, grip=None):
    objs = grasp.annotate(db.list_objects(conn, flag=flag))
    if grip:
        objs = [o for o in objs if o["grasp"]["verdict"] in grasp.FILTERS[grip]]
    groups = {}
    for o in objs:
        groups.setdefault(o.get("category") or "uncategorized", []).append(o)
    s = db.summary(conn)
    body = "".join(f'<h2>{html.escape(cat)} <span class="muted">({len(v)})</span></h2>'
                   f'<div class="grid">{"".join(_card(o) for o in v)}</div>' for cat, v in sorted(groups.items()))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>Object catalog</title>'
            f'<style>{CSS}</style></head><body><header><h1>Object catalog</h1><div class="muted">'
            f'{len(objs)} shown · {s["given"]} given · {s["found"]} found · {s["seed"]} seed · '
            f'{s["owned"]} owned · {s["no_photo"]} without photo · {s["no_model"]} without model · '
            f'generated {db.now()[:16].replace("T", " ")} UTC</div></header><main>{body or "<p>No objects.</p>"}'
            f'</main></body></html>')


def write(conn, out=None, flag=None, grip=None):
    out = out or os.path.join(config.data_dir(), "gallery.html")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(render(conn, flag, grip))
    return out
