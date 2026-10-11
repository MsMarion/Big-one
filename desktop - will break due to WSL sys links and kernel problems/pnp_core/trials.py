"""The trial log: every grab attempt in one SQLite file (/data/trials.sqlite), so runs can be compared.

    runs        one soak / session: when, which commit, scene, seed, tier, flags (speed profile, servo_error, reality,
                closed_loop, scoring)
    attempts    one pick: the object, where it truly was (sim), the nearest obstacle, the result and a failure class,
                the jaw width vs the expected one, which grasp candidate, retries, the close look's corrections, and
                the seconds per stage (detect / plan / move / look / total)
    candidates  every grasp candidate the pick considered for an attempt: closing side, yaw, clearance, wrist roll at
                pick and at place, yaw-error tolerance, joint margin, score, chosen, why rejected
    events      anything notable, from any node: calibration state changes, scene-change decisions, grip verdicts,
                staged-look back-offs, close-look rejections, recoveries (source, kind, severity, data JSON). A task's
                events carry its `task` id, the same as its attempts row
    artifacts   recordings and files that belong to a run / attempt / task: session bags, failure ring-buffer
                snapshots, scan folders (kind, path, bytes)

Writers: pick_place_node (attempts + candidates + events + ring snapshots, attached to the newest OPEN run, else its
own 'manual' run), calibration_monitor_node (events), data/grab_soak.py (runs, and the sim's truth onto each attempt),
tools/record_session.sh (artifacts). Reader: tools/trials_report.py. Several processes write at once: WAL mode.
Stdlib only. event() / artifact() never raise: logging must not break the robot.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time

DB = os.environ.get("PNP_TRIALS_DB", "/data/trials.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, started REAL, commit_sha TEXT, scene TEXT, seed INTEGER, tier TEXT, flags TEXT,
    open INTEGER DEFAULT 1, note TEXT);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY, run INTEGER, n INTEGER, t REAL, object TEXT,
    true_x REAL, true_y REAL, true_yaw REAL, obstacle_gap REAL,
    result TEXT, failure TEXT, message TEXT, jaw_width REAL, expected_width REAL, chosen INTEGER, retries INTEGER,
    corrections TEXT, s_detect REAL, s_plan REAL, s_move REAL, s_look REAL, s_total REAL);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY, run INTEGER, attempt INTEGER, idx INTEGER, closing_side TEXT, yaw REAL, clearance REAL,
    roll_pick REAL, roll_place REAL, yaw_tol REAL, joint_margin REAL, score REAL, chosen INTEGER, reject TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY, t REAL, run INTEGER, task TEXT, source TEXT, kind TEXT, severity TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY, t REAL, run INTEGER, attempt INTEGER, task TEXT, kind TEXT, path TEXT, bytes INTEGER,
    note TEXT);
CREATE INDEX IF NOT EXISTS events_task ON events (task);
"""
# columns added after the first release of a table (old databases get them on connect)
MIGRATIONS = {"attempts": {"task": "TEXT"}}

FAILURE_CLASSES = ("ok", "missed", "unexpected_width", "bumped", "blocked", "unreachable", "other")


def connect(path: str = DB) -> sqlite3.Connection:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path, timeout=10.0)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")          # several nodes write at the same time
    c.executescript(SCHEMA)
    for table, cols in MIGRATIONS.items():
        have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
        for col, typ in cols.items():
            if col not in have:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    c.commit()
    return c


def open_run(c) -> int | None:
    """The newest open run, or None (events don't create a 'manual' run)."""
    r = c.execute("SELECT id FROM runs WHERE open = 1 ORDER BY id DESC LIMIT 1").fetchone()
    return int(r["id"]) if r else None


def log_event(c, source: str, kind: str, data=None, severity: str = "info", run: int | None = None,
              task: str | None = None) -> int:
    cur = c.execute("INSERT INTO events (t, run, task, source, kind, severity, data) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (time.time(), run if run is not None else open_run(c), task, source, kind, severity,
                     json.dumps(data if data is not None else {}, default=str)))
    c.commit()
    return int(cur.lastrowid)


def _size(path: str) -> int | None:
    try:
        if os.path.isdir(path):
            return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
        return os.path.getsize(path)
    except OSError:
        return None


def add_artifact(c, kind: str, path: str, run: int | None = None, attempt: int | None = None,
                 task: str | None = None, note: str = "") -> int:
    cur = c.execute("INSERT INTO artifacts (t, run, attempt, task, kind, path, bytes, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (time.time(), run if run is not None else open_run(c), attempt, task, kind, path, _size(path),
                     note))
    c.commit()
    return int(cur.lastrowid)


def event(source: str, kind: str, data=None, severity: str = "info", run: int | None = None,
          task: str | None = None, db: str = DB) -> int | None:
    """log_event with its own connection; never raises (returns None if the log can't be written)."""
    try:
        c = connect(db)
        try:
            return log_event(c, source, kind, data, severity, run, task)
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return None


def artifact(kind: str, path: str, run: int | None = None, attempt: int | None = None, task: str | None = None,
             note: str = "", db: str = DB) -> int | None:
    """add_artifact with its own connection; never raises."""
    try:
        c = connect(db)
        try:
            return add_artifact(c, kind, path, run, attempt, task, note)
        finally:
            c.close()
    except Exception:  # noqa: BLE001
        return None


def git_commit(repo: str = "/repo") -> str:
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or "?"
    except Exception:  # noqa: BLE001
        return "?"


def start_run(c, scene: str, seed: int | None, tier: str, flags: dict, note: str = "") -> int:
    """A new OPEN run (the pick node attaches its attempts to the newest open run). Closes the others."""
    c.execute("UPDATE runs SET open = 0 WHERE open = 1")
    cur = c.execute("INSERT INTO runs (started, commit_sha, scene, seed, tier, flags, open, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    (time.time(), git_commit(), scene, seed, tier, json.dumps(flags, sort_keys=True), note))
    c.commit()
    return int(cur.lastrowid)


def close_run(c, run: int) -> None:
    c.execute("UPDATE runs SET open = 0 WHERE id = ?", (run,))
    c.commit()


def current_run(c, scene: str = "", flags: dict | None = None) -> int:
    """The newest open run; none -> a 'manual' run (picks from the page)."""
    r = c.execute("SELECT id FROM runs WHERE open = 1 ORDER BY id DESC LIMIT 1").fetchone()
    return int(r["id"]) if r else start_run(c, scene, None, "manual", flags or {})


def classify_failure(outcome: str, message: str = "") -> str:
    """PickPlace outcome name + message -> a failure class."""
    m = (message or "").lower()
    if outcome == "SUCCESS":
        return "ok"
    if outcome == "BLOCKED":
        return "blocked"
    if outcome == "UNREACHABLE":
        return "unreachable"
    if "approach blocked" in m or "landed on" in m or "bump" in m:
        return "bumped"
    if "unexpected" in m:
        return "unexpected_width"
    if outcome in ("GRASP_FAILED", "DROPPED") or "empty" in m or "slipped" in m:
        return "missed"
    if outcome == "MOTION_FAILED":
        return "bumped" if "control_failed" in m else "unreachable"
    return "other"


def add_attempt(c, run: int, **f) -> int:
    n = c.execute("SELECT COUNT(*) FROM attempts WHERE run = ?", (run,)).fetchone()[0] + 1
    if isinstance(f.get("corrections"), (list, dict)):
        f["corrections"] = json.dumps(f["corrections"])
    cols = ["run", "n", "t"] + list(f)
    cur = c.execute(f"INSERT INTO attempts ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                    [run, n, time.time()] + list(f.values()))
    c.commit()
    return int(cur.lastrowid)


def update_attempt(c, attempt: int, **f) -> None:
    if not f:
        return
    c.execute(f"UPDATE attempts SET {', '.join(f'{k} = ?' for k in f)} WHERE id = ?", list(f.values()) + [attempt])
    c.commit()


def last_attempt(c, run: int):
    return c.execute("SELECT * FROM attempts WHERE run = ? ORDER BY id DESC LIMIT 1", (run,)).fetchone()


def add_candidates(c, run: int, attempt: int, rows: list) -> None:
    keys = ("idx", "closing_side", "yaw", "clearance", "roll_pick", "roll_place", "yaw_tol", "joint_margin", "score",
            "chosen", "reject")
    c.executemany(f"INSERT INTO candidates (run, attempt, {', '.join(keys)}) VALUES (?, ?, {', '.join('?' * len(keys))})",
                  [[run, attempt] + [r.get(k) for k in keys] for r in rows])
    c.commit()
