"""pnp_core/trials.py: events + artifacts tables, WAL, the old-database migration, writers that never raise."""
import sqlite3

from pnp_core import trials


def test_wal_and_tables(tmp_path):
    c = trials.connect(str(tmp_path / "t.sqlite"))
    assert c.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    names = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"runs", "attempts", "candidates", "events", "artifacts"} <= names


def test_old_database_gets_the_task_column(tmp_path):
    db = str(tmp_path / "old.sqlite")
    old = sqlite3.connect(db)
    old.execute("CREATE TABLE attempts (id INTEGER PRIMARY KEY, run INTEGER, n INTEGER, t REAL, object TEXT)")
    old.commit()
    old.close()
    c = trials.connect(db)
    assert "task" in {r["name"] for r in c.execute("PRAGMA table_info(attempts)")}


def test_events_round_trip(tmp_path):
    c = trials.connect(str(tmp_path / "t.sqlite"))
    run = trials.start_run(c, "s", 1, "quick", {})
    eid = trials.log_event(c, "pick_place", "grip", {"width_mm": 15.7, "state": "holding"}, task="T1")
    e = c.execute("SELECT * FROM events WHERE id = ?", (eid,)).fetchone()
    assert e["run"] == run and e["task"] == "T1" and e["kind"] == "grip" and '"width_mm": 15.7' in e["data"]
    trials.close_run(c, run)
    eid2 = trials.log_event(c, "calibration_monitor", "calibration_state", {"to": "DRIFT"}, "warn")
    assert c.execute("SELECT run FROM events WHERE id = ?", (eid2,)).fetchone()[0] is None   # no run created


def test_artifact_size_and_link(tmp_path):
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "a.mcap").write_bytes(b"x" * 1000)
    db = str(tmp_path / "t.sqlite")
    c = trials.connect(db)
    run = trials.start_run(c, "s", 1, "quick", {})
    a = trials.add_attempt(c, run, object="lego_2x2", result="GRASP_FAILED", failure="missed", task="T1")
    aid = trials.artifact("ring_snapshot", str(bag), attempt=a, task="T1", db=db)
    r = c.execute("SELECT * FROM artifacts WHERE id = ?", (aid,)).fetchone()
    assert r["bytes"] == 1000 and r["attempt"] == a and r["run"] == run
    assert c.execute("SELECT task FROM attempts WHERE id = ?", (a,)).fetchone()[0] == "T1"


def test_writers_never_raise(tmp_path):
    # a directory where the database file should be: connect fails -> None, no exception
    bad = tmp_path / "is_a_dir"
    bad.mkdir()
    assert trials.event("x", "y", db=str(bad)) is None
    assert trials.artifact("bag", "/nowhere", db=str(bad)) is None
