"""SQLite object database: objects (given/found/seed), photos, 3D assets, evidence, runs, sightings,
candidates, plus the tracking tables written by objsearch.track. Files live on disk under data_dir;
rows store paths relative to it. The CLI, pipeline and dashboard all go through these helpers.

    conn = db.connect()            # creates + migrates objsearch_data/objsearch.db
"""
import contextlib
import datetime as dt
import json
import os
import re
import sqlite3

import yaml

from . import config

SCHEMA = [
    # v1
    """
CREATE TABLE objects (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT,
  manufacturer TEXT,
  mpn TEXT,
  ldraw_id TEXT,
  kicad_model TEXT,
  fcc_id TEXT,
  dims_mm TEXT,               -- JSON [l, w, h], datasheet/confirmed
  aliases TEXT,               -- JSON list
  colors TEXT,                -- JSON list
  model_hints TEXT,           -- JSON list of {source, ref, verified}
  notes TEXT,
  origin TEXT NOT NULL CHECK (origin IN ('given','found','seed')),
  owned INTEGER NOT NULL DEFAULT 0,
  include_in_search INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'draft'
    CHECK (status IN ('draft','identified','model_found','verified','rejected')),
  confidence REAL,
  primary_photo_id INTEGER,
  primary_asset_id INTEGER,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE objects_fts USING fts5(
  name, aliases, mpn, manufacturer, notes, id UNINDEXED, content='objects', content_rowid='rowid');
CREATE TRIGGER objects_ai AFTER INSERT ON objects BEGIN
  INSERT INTO objects_fts(rowid, name, aliases, mpn, manufacturer, notes, id)
  VALUES (new.rowid, new.name, new.aliases, new.mpn, new.manufacturer, new.notes, new.id);
END;
CREATE TRIGGER objects_ad AFTER DELETE ON objects BEGIN
  INSERT INTO objects_fts(objects_fts, rowid, name, aliases, mpn, manufacturer, notes, id)
  VALUES ('delete', old.rowid, old.name, old.aliases, old.mpn, old.manufacturer, old.notes, old.id);
END;
CREATE TRIGGER objects_au AFTER UPDATE ON objects BEGIN
  INSERT INTO objects_fts(objects_fts, rowid, name, aliases, mpn, manufacturer, notes, id)
  VALUES ('delete', old.rowid, old.name, old.aliases, old.mpn, old.manufacturer, old.notes, old.id);
  INSERT INTO objects_fts(rowid, name, aliases, mpn, manufacturer, notes, id)
  VALUES (new.rowid, new.name, new.aliases, new.mpn, new.manufacturer, new.notes, new.id);
END;

CREATE TABLE photos (
  id INTEGER PRIMARY KEY,
  object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('ref','rgbd','crop','render')),
  source TEXT NOT NULL CHECK (source IN ('user','run','blender')),
  view TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX photos_obj ON photos(object_id);

CREATE TABLE embeddings (
  photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
  backbone TEXT NOT NULL,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL,          -- float16
  PRIMARY KEY (photo_id, backbone)
);

CREATE TABLE assets (
  id INTEGER PRIMARY KEY,
  object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  kind TEXT NOT NULL CHECK (kind IN ('usd','obj','glb','source','thumb')),
  path TEXT NOT NULL,
  source_url TEXT, source_site TEXT, license TEXT, sha256 TEXT,
  scale_factor REAL, dims_mm TEXT, tri_count INTEGER,
  created_at TEXT NOT NULL
);
CREATE INDEX assets_obj ON assets(object_id);

CREATE TABLE evidence (
  id INTEGER PRIMARY KEY,
  object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  url TEXT NOT NULL, title TEXT, snippet TEXT,
  kind TEXT,                  -- datasheet|product|model_page|search|other
  trust TEXT,                 -- grounded|fallback|user
  created_at TEXT NOT NULL
);

CREATE TABLE runs (
  id TEXT PRIMARY KEY,        -- yyyymmdd-HHMMSS-xxxx, also the run dir name
  kind TEXT NOT NULL,         -- pipeline|demo|draft|bench
  status TEXT NOT NULL CHECK (status IN ('running','ok','failed','partial')),
  started_at TEXT NOT NULL, ended_at TEXT,
  frame_dir TEXT, provider TEXT, model TEXT,
  git_sha TEXT, config_hash TEXT, versions TEXT,
  tokens_in INTEGER DEFAULT 0, tokens_out INTEGER DEFAULT 0, cost_usd_est REAL DEFAULT 0,
  log_path TEXT, scene_usd_path TEXT, notes TEXT
);

CREATE TABLE sightings (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  instance_idx INTEGER NOT NULL,
  object_id TEXT REFERENCES objects(id) ON DELETE SET NULL,
  crop_path TEXT, mask_path TEXT,
  observed_dims_mm TEXT, ocr_text TEXT, pose TEXT,
  match_method TEXT CHECK (match_method IN ('retrieval','agent','manual')),
  score REAL,
  status TEXT NOT NULL DEFAULT 'unknown' CHECK (status IN ('matched','unknown','confirmed','rejected'))
);
CREATE INDEX sightings_obj ON sightings(object_id);

CREATE TABLE candidates (
  id INTEGER PRIMARY KEY,
  object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  sighting_id INTEGER REFERENCES sightings(id) ON DELETE SET NULL,
  source_url TEXT, license TEXT, local_path TEXT,
  scores TEXT, rank INTEGER, chosen INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);

-- tracking (objsearch.track)
CREATE TABLE steps (
  id INTEGER PRIMARY KEY,
  run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  parent_step_id INTEGER REFERENCES steps(id),
  object_id TEXT, sighting_id INTEGER,
  stage TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('running','ok','skipped','failed')),
  started_at TEXT NOT NULL, ended_at TEXT, duration_ms INTEGER,
  inputs TEXT, outputs TEXT,
  error_type TEXT, error_msg TEXT, traceback_path TEXT
);
CREATE INDEX steps_run ON steps(run_id);

CREATE TABLE llm_calls (
  id INTEGER PRIMARY KEY,
  step_id INTEGER REFERENCES steps(id) ON DELETE CASCADE,
  provider TEXT, model TEXT, request_id TEXT, stop_reason TEXT, effort TEXT,
  tokens_in INTEGER, tokens_out INTEGER, cache_read INTEGER, cache_write INTEGER,
  cost_usd_est REAL, latency_ms INTEGER, created_at TEXT NOT NULL
);
CREATE INDEX llm_calls_step ON llm_calls(step_id);

CREATE TABLE tool_calls (
  id INTEGER PRIMARY KEY,
  step_id INTEGER REFERENCES steps(id) ON DELETE CASCADE,
  llm_call_id INTEGER REFERENCES llm_calls(id),
  tool TEXT NOT NULL, args TEXT, result_summary TEXT, is_error INTEGER NOT NULL DEFAULT 0,
  latency_ms INTEGER, created_at TEXT NOT NULL
);
CREATE INDEX tool_calls_step ON tool_calls(step_id);

CREATE TABLE quota_usage (
  day TEXT NOT NULL, provider TEXT NOT NULL, model TEXT NOT NULL, kind TEXT NOT NULL,
  count INTEGER NOT NULL DEFAULT 0, tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (day, provider, model, kind)
);

CREATE TABLE http_requests (
  id INTEGER PRIMARY KEY,
  step_id INTEGER REFERENCES steps(id) ON DELETE CASCADE,
  method TEXT, url TEXT, status INTEGER, bytes INTEGER, sha256 TEXT,
  latency_ms INTEGER, cached INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL
);

CREATE TABLE object_events (
  id INTEGER PRIMARY KEY,
  object_id TEXT NOT NULL,
  actor TEXT NOT NULL CHECK (actor IN ('user','agent','pipeline')),
  field TEXT NOT NULL, old TEXT, new TEXT, reason TEXT, run_id TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX object_events_obj ON object_events(object_id);

CREATE TABLE evals (
  id INTEGER PRIMARY KEY,
  bench TEXT NOT NULL, git_sha TEXT, provider TEXT, model TEXT, config_hash TEXT,
  metrics TEXT NOT NULL, created_at TEXT NOT NULL
);
""",
    # v2: public reference images (seeding visuals, with license/attribution) + collection check
    """
CREATE TABLE ref_images (
  id INTEGER PRIMARY KEY,
  object_id TEXT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
  path TEXT NOT NULL,              -- data-relative jpg (thumb: .thumb.jpg)
  title TEXT,
  provider TEXT,                   -- e.g. openverse/flickr, openverse/wikimedia
  provider_id TEXT,
  landing_url TEXT,                -- page to credit/link
  image_url TEXT,
  creator TEXT, creator_url TEXT,
  license TEXT, license_version TEXT, license_url TEXT,
  attribution TEXT,
  query TEXT,
  rank INTEGER,
  chosen INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE INDEX ref_images_obj ON ref_images(object_id);
ALTER TABLE objects ADD COLUMN ref_query TEXT;
ALTER TABLE objects ADD COLUMN collection_checked_at TEXT;
""",
    # v3: how the object rests for grasping (objsearch/grasp.py): flat (default) | upright
    """
ALTER TABLE objects ADD COLUMN grasp_pose TEXT;
""",
    # v4: internal 3D parts library (objsearch.parts): what board builds can use, and what they still lack
    """
CREATE TABLE parts3d (
  key TEXT PRIMARY KEY,           -- KiCad model path without extension (Lib.3dshapes/Name) or 'eagle:<footprint>'
  kind TEXT NOT NULL CHECK (kind IN ('kicad_model','footprint')),
  status TEXT NOT NULL CHECK (status IN ('real','approximate','missing','unmapped')),
  path TEXT,                      -- data-relative aligned STEP used for builds (real or approximate)
  source_url TEXT, license TEXT, notes TEXT,
  transform TEXT,                 -- JSON 3x4 row-major matrix applied to the source to align it
  dims_mm TEXT,                   -- JSON [x, y, z] of the aligned part
  seen_in TEXT,                   -- JSON list of object ids whose models use / lack this part
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
""",
    # v5: every object gets a size (and says where it came from), physics for the sim-ready USD, downloaded docs
    """
-- dims_source: catalog|datasheet|measured|photo_estimate|category_default
ALTER TABLE objects ADD COLUMN dims_source TEXT;
-- physics: JSON {mass_g, mass_source, materials, static_friction, restitution, source, researched_at, ...}
ALTER TABLE objects ADD COLUMN physics TEXT;
-- local_path: data-relative copy of a downloaded document (datasheet PDF)
ALTER TABLE evidence ADD COLUMN local_path TEXT;
""",
]

JSON_COLS = {"physics", "dims_mm", "aliases", "colors", "model_hints", "versions", "observed_dims_mm", "pose", "transform", "seen_in", "model_scores",
             "scores", "inputs", "outputs", "args", "old", "new", "metrics"}
EDITABLE = ("name", "category", "manufacturer", "mpn", "ldraw_id", "kicad_model", "fcc_id", "dims_mm",
            "aliases", "colors", "model_hints", "notes", "origin", "owned", "include_in_search", "status",
            "confidence", "primary_photo_id", "primary_asset_id", "ref_query", "collection_checked_at", "grasp_pose",
            "dims_source", "physics")
SEED_FIELDS = ("name", "category", "manufacturer", "mpn", "ldraw_id", "kicad_model", "fcc_id", "dims_mm",
               "aliases", "colors", "model_hints", "notes", "origin", "owned", "include_in_search", "status",
               "confidence", "ref_query", "collection_checked_at", "grasp_pose", "dims_source", "physics")
NEEDS_REVIEW = ("draft", "model_found")  # awaiting the user: confirm identity / approve a model


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def connect(path=None):
    path = path or config.db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    migrate(conn)
    return conn


def migrate(conn):
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, sql in enumerate(SCHEMA[v:], start=v + 1):
        with tx(conn):
            for stmt in _split(sql):
                conn.execute(stmt)
            conn.execute(f"PRAGMA user_version={i}")


def _split(sql):
    """Split a schema script into statements (executescript would commit our transaction)."""
    out, buf = [], []
    for line in sql.splitlines():
        if line.strip().startswith("--") and not buf:
            continue
        buf.append(line)
        joined = "\n".join(buf).strip()
        if joined.endswith(";") and sqlite3.complete_statement(joined):
            out.append(joined)
            buf = []
    rest = [ln for ln in buf if ln.strip() and not ln.strip().startswith("--")]
    if rest:  # e.g. a trailing comment after ';' hides the statement end: fail loudly, never skip SQL
        raise ValueError(f"unterminated schema statement: {' '.join(rest)[:200]}")
    return out


@contextlib.contextmanager
def tx(conn):
    """Explicit transaction on an autocommit connection; nests by becoming a no-op."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def enc(v):
    return None if v is None else json.dumps(v, ensure_ascii=False)


def row(r):
    if r is None:
        return None
    d = dict(r)
    for k in JSON_COLS & d.keys():
        if isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except ValueError:
                pass
    return d


def rows(cur):
    return [row(r) for r in cur.fetchall()]


def slugify(text):
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:60] or "object"


def unique_id(conn, base):
    oid, i = slugify(base), 2
    while conn.execute("SELECT 1 FROM objects WHERE id=?", (oid,)).fetchone():
        oid, i = f"{slugify(base)}-{i}", i + 1
    return oid


# ---------------------------------------------------------------- objects

def create_object(conn, fields, actor="user", reason=None, run_id=None, oid=None):
    f = {k: v for k, v in fields.items() if k in EDITABLE}
    if not f.get("name"):
        raise ValueError("name is required")
    f.setdefault("origin", "given")
    f.setdefault("status", "draft")
    for k in ("owned", "include_in_search"):
        f[k] = int(bool(f.get(k, f["origin"] == "given")))
    oid = oid or unique_id(conn, f["name"])
    t = now()
    cols = ["id", *f.keys(), "created_at", "updated_at"]
    vals = [oid, *[enc(v) if k in JSON_COLS else v for k, v in f.items()], t, t]
    with tx(conn):
        conn.execute(f"INSERT INTO objects ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", vals)
        log_event(conn, oid, actor, "*created*", None, {"origin": f["origin"], "status": f["status"]},
                  reason, run_id)
    return oid


def get_object(conn, oid):
    return row(conn.execute("SELECT * FROM objects WHERE id=?", (oid,)).fetchone())


def update_object(conn, oid, changes, actor="user", reason=None, run_id=None):
    """Apply field changes and write one object_events row per changed field."""
    old = get_object(conn, oid)
    if old is None:
        raise KeyError(oid)
    bad = set(changes) - set(EDITABLE)
    if bad:
        raise ValueError(f"not editable: {sorted(bad)}")
    diff = {}
    for k, v in changes.items():
        if k in ("owned", "include_in_search"):
            v = int(bool(v))
        if old.get(k) != v:
            diff[k] = v
    if not diff:
        return old
    with tx(conn):
        sets = ", ".join(f"{k}=?" for k in diff) + ", updated_at=?"
        conn.execute(f"UPDATE objects SET {sets} WHERE id=?",
                     [*(enc(v) if k in JSON_COLS else v for k, v in diff.items()), now(), oid])
        for k, v in diff.items():
            log_event(conn, oid, actor, k, old.get(k), v, reason, run_id)
    return get_object(conn, oid)


def log_event(conn, oid, actor, field, old, new, reason=None, run_id=None):
    conn.execute("INSERT INTO object_events (object_id, actor, field, old, new, reason, run_id, created_at) "
                 "VALUES (?,?,?,?,?,?,?,?)", (oid, actor, field, enc(old), enc(new), reason, run_id, now()))


def fts_query(q):
    """User text -> safe FTS5 query: every token as a quoted prefix term (AND)."""
    toks = re.findall(r"[\w.]+", q or "")
    return " ".join(f'"{t}"*' for t in toks)


_LIST_SQL = """
SELECT o.*,
  (SELECT COUNT(*) FROM photos p WHERE p.object_id=o.id AND p.source='user') AS photo_count,
  (SELECT COUNT(*) FROM assets a WHERE a.object_id=o.id AND a.kind IN ('usd','glb','obj')) AS asset_count,
  (SELECT COUNT(*) FROM sightings s WHERE s.object_id=o.id) AS sighting_count,
  COALESCE((SELECT path FROM photos WHERE id=o.primary_photo_id),
           (SELECT path FROM photos p WHERE p.object_id=o.id AND p.kind='ref' ORDER BY p.id LIMIT 1),
           (SELECT path FROM photos p WHERE p.object_id=o.id ORDER BY p.id LIMIT 1)) AS photo_path,
  (SELECT path FROM assets a WHERE a.object_id=o.id AND a.kind='thumb' ORDER BY a.id DESC LIMIT 1) AS thumb_path,
  (SELECT path FROM assets a WHERE a.object_id=o.id AND a.kind='glb' ORDER BY a.id DESC LIMIT 1) AS glb_path,
  (SELECT path FROM ref_images r WHERE r.object_id=o.id ORDER BY r.chosen DESC, r.rank LIMIT 1) AS ref_path,
  (SELECT attribution FROM ref_images r WHERE r.object_id=o.id ORDER BY r.chosen DESC, r.rank LIMIT 1) AS ref_attribution,
  (SELECT scores FROM candidates c WHERE c.object_id=o.id AND c.chosen=1 ORDER BY c.id DESC LIMIT 1) AS model_scores
FROM objects o
"""

FLAGS = {
    "given": "o.origin='given'",
    "found": "o.origin='found'",
    "seed": "o.origin='seed'",
    "owned": "o.owned=1",
    "needs_review": "o.status IN ('draft','model_found')",
    "no_photo": "NOT EXISTS (SELECT 1 FROM photos p WHERE p.object_id=o.id AND p.source='user')",
    "no_model": "NOT EXISTS (SELECT 1 FROM assets a WHERE a.object_id=o.id AND a.kind IN ('usd','glb','obj'))",
    "verified": "o.status='verified'",
    "unchecked": "o.collection_checked_at IS NULL",
    "no_ref": "NOT EXISTS (SELECT 1 FROM ref_images r WHERE r.object_id=o.id)",
}


def list_objects(conn, flag=None, category=None, q=None, status=None, include_rejected=False, limit=1000):
    where, args = [], []
    if flag:
        if flag not in FLAGS:
            raise ValueError(f"unknown filter {flag}")
        where.append(FLAGS[flag])
    if category:
        where.append("o.category=?")
        args.append(category)
    if status:
        where.append("o.status=?")
        args.append(status)
    elif not include_rejected:
        where.append("o.status!='rejected'")
    if q and fts_query(q):
        where.append("o.rowid IN (SELECT rowid FROM objects_fts WHERE objects_fts MATCH ?)")
        args.append(fts_query(q))
    sql = _LIST_SQL + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY o.category, o.name LIMIT ?"
    return rows(conn.execute(sql, [*args, limit]))


def search(conn, text, limit=10):
    """FTS search ranked by bm25 (used by the agent's catalog_lookup tool)."""
    fq = fts_query(text)
    if not fq:
        return []
    return rows(conn.execute(
        "SELECT o.id, o.name, o.category, o.mpn, o.dims_mm, o.aliases, o.status, bm25(objects_fts) AS rank "
        "FROM objects_fts JOIN objects o ON o.rowid=objects_fts.rowid WHERE objects_fts MATCH ? "
        "AND o.status!='rejected' ORDER BY rank LIMIT ?", (fq, limit)))


def summary(conn):
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    base = "FROM objects o WHERE o.status!='rejected'"
    out = {k: one(f"SELECT COUNT(*) {base} AND {FLAGS[k]}") for k in FLAGS}
    out["all"] = one(f"SELECT COUNT(*) {base}")
    out["categories"] = [dict(r) for r in conn.execute(
        f"SELECT category, COUNT(*) AS n {base} GROUP BY category ORDER BY category")]
    out["sightings"] = one("SELECT COUNT(*) FROM sightings")
    out["runs"] = one("SELECT COUNT(*) FROM runs")
    return out


def get_object_full(conn, oid):
    o = row(conn.execute(_LIST_SQL + " WHERE o.id=?", (oid,)).fetchone())
    if o is None:
        return None
    q = lambda sql: rows(conn.execute(sql, (oid,)))  # noqa: E731
    o["photos"] = q("SELECT * FROM photos WHERE object_id=? ORDER BY source='user' DESC, id")
    o["assets"] = q("SELECT * FROM assets WHERE object_id=? ORDER BY id")
    o["evidence"] = q("SELECT * FROM evidence WHERE object_id=? ORDER BY id")
    o["sightings"] = q("SELECT s.*, r.started_at FROM sightings s JOIN runs r ON r.id=s.run_id "
                       "WHERE s.object_id=? ORDER BY r.started_at DESC LIMIT 200")
    o["candidates"] = q("SELECT * FROM candidates WHERE object_id=? ORDER BY chosen DESC, rank, id")
    o["events"] = q("SELECT * FROM object_events WHERE object_id=? ORDER BY id DESC LIMIT 200")
    o["ref_images"] = q("SELECT * FROM ref_images WHERE object_id=? ORDER BY chosen DESC, rank, id")
    return o


# ---------------------------------------------------------------- children

def _insert(conn, table, fields):
    fields = {k: enc(v) if k in JSON_COLS else v for k, v in fields.items()}
    cols = list(fields)
    cur = conn.execute(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                       list(fields.values()))
    return cur.lastrowid


def add_photo(conn, object_id, path, kind="ref", source="user", view=None, actor="user"):
    with tx(conn):
        pid = _insert(conn, "photos", dict(object_id=object_id, path=path, kind=kind, source=source,
                                           view=view, created_at=now()))
        o = get_object(conn, object_id)
        if o["primary_photo_id"] is None and source == "user":
            conn.execute("UPDATE objects SET primary_photo_id=?, updated_at=? WHERE id=?", (pid, now(), object_id))
        log_event(conn, object_id, actor, "photo_added", None, {"id": pid, "path": path, "kind": kind})
    return pid


def add_asset(conn, object_id, kind, path, actor="pipeline", **meta):
    with tx(conn):
        aid = _insert(conn, "assets", dict(object_id=object_id, kind=kind, path=path, created_at=now(), **meta))
        log_event(conn, object_id, actor, "asset_added", None, {"id": aid, "kind": kind, "path": path})
    return aid


def add_evidence(conn, object_id, url, title=None, snippet=None, kind="other", trust=None, local_path=None):
    return _insert(conn, "evidence", dict(object_id=object_id, url=url, title=title, snippet=snippet, local_path=local_path,
                                          kind=kind, trust=trust, created_at=now()))


def add_ref_image(conn, object_id, **fields):
    return _insert(conn, "ref_images", dict(object_id=object_id, created_at=now(), **fields))


def choose_ref_image(conn, object_id, ref_id, actor="user"):
    with tx(conn):
        n = conn.execute("UPDATE ref_images SET chosen=(id=?) WHERE object_id=?", (ref_id, object_id)).rowcount
        log_event(conn, object_id, actor, "ref_image_chosen", None, {"id": ref_id})
    return n


def delete_ref_image(conn, object_id, ref_id, actor="user"):
    """Remove a reference image row (caller deletes files); the next best becomes chosen. -> row or None."""
    r = row(conn.execute("SELECT * FROM ref_images WHERE id=? AND object_id=?", (ref_id, object_id)).fetchone())
    if r is None:
        return None
    with tx(conn):
        conn.execute("DELETE FROM ref_images WHERE id=?", (ref_id,))
        log_event(conn, object_id, actor, "ref_image_removed", {"id": ref_id, "title": r["title"]}, None)
        if r["chosen"]:
            nxt = conn.execute("SELECT id FROM ref_images WHERE object_id=? ORDER BY rank LIMIT 1", (object_id,)).fetchone()
            if nxt:
                conn.execute("UPDATE ref_images SET chosen=1 WHERE id=?", (nxt[0],))
    return r


def add_sighting(conn, run_id, instance_idx, **fields):
    return _insert(conn, "sightings", dict(run_id=run_id, instance_idx=instance_idx, **fields))


def add_candidate(conn, object_id, **fields):
    """One row per (object, source_url): a repeat sighting refreshes scores/rank instead of duplicating."""
    url = fields.get("source_url")
    old = url and conn.execute("SELECT id FROM candidates WHERE object_id=? AND source_url=?", (object_id, url)).fetchone()
    if old:
        upd = {k: v for k, v in fields.items() if k in ("scores", "rank", "license", "local_path", "sighting_id")}
        if upd:
            conn.execute(f"UPDATE candidates SET {', '.join(f'{k}=?' for k in upd)} WHERE id=?",
                         [*(enc(v) if k in JSON_COLS else v for k, v in upd.items()), old[0]])
        return old[0]
    return _insert(conn, "candidates", dict(object_id=object_id, created_at=now(), **fields))


# ---------------------------------------------------------------- runs / activity (dashboard reads)

def list_runs(conn, limit=100):
    return rows(conn.execute(
        "SELECT r.*, (SELECT COUNT(*) FROM steps s WHERE s.run_id=r.id) AS n_steps, "
        "(SELECT COUNT(*) FROM steps s WHERE s.run_id=r.id AND s.status='failed') AS n_failed, "
        "(SELECT COUNT(*) FROM sightings s WHERE s.run_id=r.id) AS n_sightings "
        "FROM runs r ORDER BY r.started_at DESC LIMIT ?", (limit,)))


def get_run(conn, run_id):
    r = row(conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())
    if r is None:
        return None
    q = lambda sql: rows(conn.execute(sql, (run_id,)))  # noqa: E731
    r["steps"] = q("SELECT * FROM steps WHERE run_id=? ORDER BY id")
    r["llm_calls"] = q("SELECT l.* FROM llm_calls l JOIN steps s ON s.id=l.step_id WHERE s.run_id=? ORDER BY l.id")
    r["tool_calls"] = q("SELECT t.* FROM tool_calls t JOIN steps s ON s.id=t.step_id WHERE s.run_id=? ORDER BY t.id")
    r["http_requests"] = q("SELECT h.* FROM http_requests h JOIN steps s ON s.id=h.step_id "
                           "WHERE s.run_id=? ORDER BY h.id")
    r["sightings"] = q("SELECT s.*, o.name AS object_name FROM sightings s LEFT JOIN objects o ON o.id=s.object_id "
                       "WHERE s.run_id=? ORDER BY s.instance_idx")
    return r


def activity(conn, limit=50):
    ev = rows(conn.execute("SELECT e.*, o.name AS object_name FROM object_events e "
                           "LEFT JOIN objects o ON o.id=e.object_id ORDER BY e.id DESC LIMIT ?", (limit,)))
    failed = rows(conn.execute("SELECT id, run_id, stage, object_id, error_type, error_msg, started_at FROM steps "
                               "WHERE status='failed' ORDER BY id DESC LIMIT ?", (limit,)))
    return {"events": ev, "failed_steps": failed}


def usage_totals(conn, days=14):
    return {
        "by_day": rows(conn.execute(
            "SELECT substr(created_at,1,10) AS day, provider, COUNT(*) AS calls, SUM(tokens_in) AS tokens_in, "
            "SUM(tokens_out) AS tokens_out, ROUND(SUM(cost_usd_est),4) AS cost FROM llm_calls "
            "GROUP BY day, provider ORDER BY day DESC LIMIT ?", (days * 4,))),
        "by_stage": rows(conn.execute(
            "SELECT stage, COUNT(*) AS n, SUM(status='failed') AS failed, SUM(status='skipped') AS skipped, "
            "CAST(AVG(duration_ms) AS INTEGER) AS avg_ms FROM steps GROUP BY stage ORDER BY n DESC")),
    }


# ---------------------------------------------------------------- seed import / export

_CHILD_EXPORT = {
    "photos": ("path", "kind", "source", "view"),
    "assets": ("kind", "path", "source_url", "source_site", "license", "sha256", "scale_factor", "dims_mm",
               "tri_count"),
    "evidence": ("url", "title", "snippet", "kind", "trust", "local_path"),
    "ref_images": ("path", "title", "provider", "provider_id", "landing_url", "image_url", "creator", "creator_url",
                   "license", "license_version", "license_url", "attribution", "query", "rank", "chosen"),
}


def export_seed(conn, path=None):
    """All non-rejected objects with their photos/assets/evidence -> YAML (sorted, None-free)."""
    out = []
    for o in rows(conn.execute("SELECT * FROM objects WHERE status!='rejected' ORDER BY id")):
        e = {"id": o["id"], **{k: o[k] for k in SEED_FIELDS if o.get(k) not in (None, [], "")}}
        for k in ("owned", "include_in_search"):
            e[k] = bool(o[k])
        for table, cols in _CHILD_EXPORT.items():
            kids = rows(conn.execute(f"SELECT * FROM {table} WHERE object_id=? ORDER BY id", (o["id"],)))
            kids = [{c: k[c] for c in cols if k.get(c) is not None} for k in kids]
            if kids:
                e[table] = kids
        if o["primary_photo_id"]:
            p = conn.execute("SELECT path FROM photos WHERE id=?", (o["primary_photo_id"],)).fetchone()
            if p:
                e["primary_photo"] = p[0]
        out.append(e)
    text = yaml.safe_dump({"version": 1, "objects": out}, sort_keys=False, allow_unicode=True, width=110)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def import_seed(conn, path, update=False, actor="user"):
    """Insert objects from YAML. Existing ids are skipped unless update=True. Returns counts."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    n = {"created": 0, "updated": 0, "skipped": 0}
    with tx(conn):
        for e in data.get("objects", []):
            fields = {k: e[k] for k in SEED_FIELDS if k in e}
            fields.setdefault("origin", "seed")
            fields.setdefault("status", "identified" if fields["origin"] == "seed" else "draft")
            oid = e.get("id") or slugify(fields["name"])
            if get_object(conn, oid):
                if update:
                    update_object(conn, oid, fields, actor, reason=f"import {os.path.basename(path)}")
                    n["updated"] += 1
                else:
                    n["skipped"] += 1
                continue
            create_object(conn, fields, actor, reason=f"import {os.path.basename(path)}", oid=oid)
            for p in e.get("photos", []):
                add_photo(conn, oid, p["path"], p.get("kind", "ref"), p.get("source", "user"), p.get("view"), actor)
            for a in e.get("assets", []):
                add_asset(conn, oid, a.pop("kind"), a.pop("path"), actor, **a)
            for ev in e.get("evidence", []):
                add_evidence(conn, oid, **ev)
            for ri in e.get("ref_images", []):
                add_ref_image(conn, oid, **ri)
            if e.get("primary_photo"):
                p = conn.execute("SELECT id FROM photos WHERE object_id=? AND path=?",
                                 (oid, e["primary_photo"])).fetchone()
                if p:
                    conn.execute("UPDATE objects SET primary_photo_id=? WHERE id=?", (p[0], oid))
            n["created"] += 1
    return n


def reset_inventory(conn):
    """'Clear my inventory': forget what you own and everything observed/built for it, but keep knowledge
    that makes rebuilding fast - catalog entries (incl. learned model sources), the 3D parts library,
    reference images, the model build cache and quota history. Returns counts."""
    import shutil
    n = {}
    with tx(conn):
        n["runs"] = conn.execute("DELETE FROM runs").rowcount            # cascades steps, calls, sightings
        n["photos"] = conn.execute("DELETE FROM photos").rowcount
        n["assets"] = conn.execute("DELETE FROM assets").rowcount
        n["candidates"] = conn.execute("DELETE FROM candidates").rowcount
        added = [r[0] for r in conn.execute("SELECT id FROM objects WHERE origin IN ('given','found') "
                                            "AND id NOT IN (SELECT id FROM objects WHERE origin='seed')")]
        seeds = {r[0] for r in conn.execute("SELECT id FROM objects WHERE origin='seed'")}
        # objects that started as seeds were converted to 'given' when owned: find them via their creation event
        was_seed = {r[0] for r in conn.execute(
            "SELECT object_id FROM object_events WHERE field='*created*' AND new LIKE '%\"origin\": \"seed\"%'")}
        drop = [i for i in added if i not in was_seed]
        for oid in drop:
            conn.execute("DELETE FROM objects WHERE id=?", (oid,))
        n["objects_removed"] = len(drop)
        n["catalog_reset"] = conn.execute(
            "UPDATE objects SET origin='seed', owned=0, include_in_search=0, collection_checked_at=NULL, "
            "primary_photo_id=NULL, primary_asset_id=NULL, status=CASE WHEN status='rejected' THEN status "
            "ELSE 'identified' END, confidence=NULL, updated_at=? WHERE id IN (SELECT object_id FROM object_events "
            "WHERE field='*created*' AND new LIKE '%\"origin\": \"seed\"%') OR origin='seed'", (now(),)).rowcount
        conn.execute("DELETE FROM object_events WHERE object_id NOT IN (SELECT id FROM objects)")
        conn.execute("DELETE FROM evidence WHERE object_id NOT IN (SELECT id FROM objects)")
        conn.execute("DELETE FROM ref_images WHERE object_id NOT IN (SELECT id FROM objects)")
    for d in ("photos", "assets", "runs", "quarantine"):
        shutil.rmtree(os.path.join(config.data_dir(), d), ignore_errors=True)
    return n
