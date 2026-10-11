import json
import os

import pytest

from objsearch import config, db, demo, track


def _log_records(conn, run_id):
    rel = conn.execute("SELECT log_path FROM runs WHERE id=?", (run_id,)).fetchone()[0]
    with open(os.path.join(config.data_dir(), rel), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _check_closed(r):
    assert r["status"] != "running" and r["ended_at"]
    assert all(s["status"] != "running" and s["ended_at"] and s["duration_ms"] is not None for s in r["steps"])


def test_demo_ok_run_is_fully_tracked(seeded):
    rid = demo.run_demo(seeded)
    r = db.get_run(seeded, rid)
    _check_closed(r)
    assert r["status"] == "ok"
    stages = [s["stage"] for s in r["steps"]]
    assert stages[0] == "observe" and stages.count("retrieve") == 3 and "identify" in stages and "acquire" in stages
    by = {s["id"]: s for s in r["steps"]}
    nested = [s for s in r["steps"] if s["parent_step_id"]]
    assert {s["stage"] for s in nested} == {"download", "convert", "compare"}
    assert all(by[s["parent_step_id"]]["stage"] == "acquire" for s in nested)
    assert len(r["llm_calls"]) == 9 and len(r["tool_calls"]) == 7 and r["http_requests"]
    assert r["tokens_in"] > 0 and r["config_hash"]
    # sightings: two matched by retrieval, one by the agent (found object)
    methods = sorted(s["match_method"] for s in r["sightings"])
    assert methods == ["agent", "retrieval", "retrieval"]
    found = db.get_object(seeded, "demo-generic-1080p-usb-webcam-demo")
    assert found["origin"] == "found" and found["status"] == "model_found"
    # transcript saved for agent steps
    ident = next(s for s in r["steps"] if s["stage"] == "identify")
    assert os.path.exists(os.path.join(config.data_dir(), ident["outputs"]["transcript"]))
    # JSONL log: every record belongs to this run; step ids match the DB both ways
    recs = _log_records(seeded, rid)
    assert recs and all(x["run_id"] == rid for x in recs)
    logged = {x["step_id"] for x in recs if x["step_id"]}
    assert logged == set(by)


def test_fail_mid_acquire(seeded):
    rid = demo.run_demo(seeded, fail_at="acquire")
    r = db.get_run(seeded, rid)
    _check_closed(r)
    assert r["status"] == "partial"
    acq = next(s for s in r["steps"] if s["stage"] == "acquire")
    assert acq["status"] == "failed" and acq["error_type"] == "ConnectionError"
    tb = os.path.join(config.data_dir(), acq["traceback_path"])
    assert "ConnectionError" in open(tb, encoding="utf-8").read()
    assert any(c["step_id"] == acq["id"] for c in r["llm_calls"])
    assert any(c["step_id"] == acq["id"] for c in r["tool_calls"])
    assert next(s for s in r["steps"] if s["stage"] == "assemble")["status"] == "skipped"  # run continued
    errs = [x for x in _log_records(seeded, rid) if x["level"] == "ERROR"]
    assert errs and errs[0]["step_id"] == acq["id"]
    assert db.activity(seeded)["failed_steps"][0]["id"] == acq["id"]


def test_tool_failure_in_nested_step_is_recovered(seeded):
    r = db.get_run(seeded, demo.run_demo(seeded, fail_at="convert"))
    _check_closed(r)
    conv = next(s for s in r["steps"] if s["stage"] == "convert")
    assert conv["status"] == "failed" and conv["error_type"] == "ValueError"
    assert next(s for s in r["steps"] if s["stage"] == "acquire")["status"] == "ok"
    assert any(t["tool"] == "convert" and t["is_error"] for t in r["tool_calls"])
    assert r["status"] == "partial"


def test_secrets_are_redacted(conn, monkeypatch):
    key = "AIzaSyD" + "x" * 32
    monkeypatch.setenv("GEMINI_API_KEY", key)
    with track.run(conn, "test") as r:
        with track.step("probe", inputs={"api_key": "hunter2hunter2", "url": f"https://h/?key={key}"}):
            track.record_tool_call("web_fetch", {"url": f"https://h/x?key={key}&q=1", "token": "abc"},
                                   f"echo {key}")
            track.record_http("GET", f"https://h/x?key={key}")
            track.log.info(f"calling with {key} and Authorization: Bearer sk-ant-api03-abcdefghijklmnop")
        rid = r.id
    dump = json.dumps(db.get_run(conn, rid)) + json.dumps(_log_records(conn, rid))
    assert key not in dump and "hunter2" not in dump and "sk-ant-api03" not in dump
    assert "***" in dump


def test_quota_and_skips(conn):
    with track.run(conn, "test") as r:
        track.bump_quota(conn, "gemini", "gemini-2.5-flash", "grounded_search", 2)
        assert track.check_quota(conn, "grounded_search", 3) == 2
        track.bump_quota(conn, "gemini", "gemini-2.5-flash", "grounded_search")
        with track.step("search") as s:
            track.check_quota(conn, "grounded_search", 3)
        with track.step("x") as s2:
            s2.skip("nothing to do")
        rid = r.id
    steps = db.get_run(conn, rid)["steps"]
    assert [(s["stage"], s["status"]) for s in steps] == [("search", "skipped"), ("x", "skipped")]
    assert "quota reached" in steps[0]["outputs"]["skip_reason"]
    assert track.quota_status(conn)["grounded_search"]["used"] == 3


def test_step_requires_run_and_failed_run(conn):
    with pytest.raises(RuntimeError):
        with track.step("orphan"):
            pass
    with pytest.raises(KeyError):
        with track.run(conn, "test") as r:
            with track.step("boom"):
                raise KeyError("x")
    assert db.get_run(conn, r.id)["status"] == "failed"
    assert track.current_run() is None


def test_runs_nest(conn):
    with track.run(conn, "outer") as outer:
        with track.step("before"):
            pass
        with track.run(conn, "inner") as inner:
            with track.step("inside"):
                pass
        assert track.current_run() is outer  # restored after the inner run
        with track.step("after"):
            pass
    assert [s["stage"] for s in db.get_run(conn, outer.id)["steps"]] == ["before", "after"]
    assert [s["stage"] for s in db.get_run(conn, inner.id)["steps"]] == ["inside"]
    assert track.current_run() is None
