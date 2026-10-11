"""Run/step tracking and structured logging.

Every pipeline action happens inside a run and a (possibly nested) step:

    with track.run(conn, "pipeline", provider="gemini", frame_dir=d) as r:
        with track.step("observe", inputs={"frame": d}) as s:
            ...
            s.set(instances=3)

Each step is a `steps` row (running -> ok/skipped/failed, duration, inputs/outputs, error + traceback
file). LLM calls, tool calls and HTTP requests made inside a step attach to it automatically through
contextvars. Every log record carries run_id/step_id/object_id/stage and goes to the run's
`run.log.jsonl` (plus a rotating global log), with secrets redacted.
"""
import contextlib
import contextvars
import datetime as dt
import functools
import importlib.metadata
import json
import logging
import logging.handlers
import os
import re
import secrets
import subprocess
import time
import traceback

from . import config, db

log = logging.getLogger("objsearch")

_run = contextvars.ContextVar("objsearch_run", default=None)
_step = contextvars.ContextVar("objsearch_step", default=None)


class SkipStep(Exception):
    """Raise (or call Step.skip) to end a step with status 'skipped'."""


class QuotaExceeded(Exception):
    """A local free-tier quota is used up; steps record this as 'skipped', not a crash."""


# ---------------------------------------------------------------- redaction

_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), "***"),
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"), "***"),
    (re.compile(r"(?i)\b(key|token|api_key|apikey|access_token|authorization)=([^&\s\"']+)"), r"\1=***"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1***"),
]
_SECRET_KEYS = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization|cookie)")


def redact(v):
    if isinstance(v, str):
        for s in config.secrets():
            v = v.replace(s, "***")
        for pat, rep in _PATTERNS:
            v = pat.sub(rep, v)
        return v
    if isinstance(v, dict):
        return {k: "***" if isinstance(k, str) and _SECRET_KEYS.search(k) else redact(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [redact(x) for x in v]
    return v


# ---------------------------------------------------------------- logging

class _ContextFilter(logging.Filter):
    def filter(self, record):
        r, s = _run.get(), _step.get()
        record.run_id = r.id if r else None
        record.step_id = s.id if s else None
        record.stage = s.stage if s else ("run" if r else "-")
        record.object_id = s.object_id if s else None
        return True


_CTX = _ContextFilter()


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": dt.datetime.fromtimestamp(record.created, dt.timezone.utc).isoformat(timespec="milliseconds"),
             "level": record.levelname, "logger": record.name, "msg": redact(record.getMessage()),
             "run_id": getattr(record, "run_id", None), "step_id": getattr(record, "step_id", None),
             "stage": getattr(record, "stage", None), "object_id": getattr(record, "object_id", None)}
        if getattr(record, "data", None) is not None:
            d["data"] = redact(record.data)
        if record.exc_info:
            d["exc"] = redact(self.formatException(record.exc_info))
        return json.dumps(d, ensure_ascii=False, default=str)


class _RunFilter(logging.Filter):
    def __init__(self, run_id):
        super().__init__()
        self.run_id = run_id

    def filter(self, record):
        return getattr(record, "run_id", None) == self.run_id


@functools.lru_cache(maxsize=1)
def setup_logging(level=logging.INFO, console=True):
    """Idempotent: console (human) + rotating JSON log under data_dir/logs."""
    log.setLevel(logging.DEBUG)
    log.propagate = False
    os.makedirs(os.path.join(config.data_dir(), "logs"), exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(os.path.join(config.data_dir(), "logs", "objsearch.log"),
                                              maxBytes=5_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(_JsonFormatter())
    fh.addFilter(_CTX)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.addFilter(_CTX)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(stage)s] %(message)s", "%H:%M:%S"))
        log.addHandler(ch)
    return True


# ---------------------------------------------------------------- reproducibility

@functools.lru_cache(maxsize=1)
def git_sha():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, capture_output=True,
                             text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "objsearch"], cwd=config.ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("+dirty" if dirty else "") if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def versions():
    out = {}
    for p in ("google-genai", "anthropic", "bpy", "trimesh", "cascadio", "transformers", "torch"):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            pass
    return out


# ---------------------------------------------------------------- runs

class Run:
    def __init__(self, conn, run_id, run_dir, handler):
        self.conn, self.id, self.dir, self._handler = conn, run_id, run_dir, handler

    def path(self, *parts):
        """Absolute path inside the run dir; db.rel() turns it into a data_dir-relative path."""
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p


def rel(path):
    """data_dir-relative POSIX path, as stored in the DB and served at /objdata/."""
    return os.path.relpath(path, config.data_dir()).replace(os.sep, "/")


def start_run(conn, kind, provider=None, model=None, frame_dir=None, notes=None):
    setup_logging()
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
    run_dir = os.path.join(config.data_dir(), "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log.jsonl")
    with db.tx(conn):
        db._insert(conn, "runs", dict(
            id=run_id, kind=kind, status="running", started_at=db.now(), frame_dir=frame_dir, provider=provider,
            model=model, git_sha=git_sha(), config_hash=config.config_hash(), versions=versions(),
            log_path=rel(log_path), notes=notes))
    h = logging.FileHandler(log_path, encoding="utf-8")
    h.setFormatter(_JsonFormatter())
    h.addFilter(_CTX)  # handler-level so records from child loggers (objsearch.*) get ids too
    h.addFilter(_RunFilter(run_id))
    log.addHandler(h)
    r = Run(conn, run_id, run_dir, h)
    r.prev, r.prev_step = _run.get(), _step.get()  # runs nest (e.g. a parts job that rebuilds models)
    _run.set(r)
    _step.set(None)
    log.info("run started", extra={"data": {"kind": kind, "provider": provider, "model": model}})
    return r


def end_run(status=None):
    r = _run.get()
    if r is None:
        return None
    c = r.conn
    failed = c.execute("SELECT COUNT(*) FROM steps WHERE run_id=? AND status='failed'", (r.id,)).fetchone()[0]
    if status is None:
        status = "partial" if failed else "ok"
    tot = c.execute("SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), COALESCE(SUM(cost_usd_est),0) "
                    "FROM llm_calls l JOIN steps s ON s.id=l.step_id WHERE s.run_id=?", (r.id,)).fetchone()
    c.execute("UPDATE runs SET status=?, ended_at=?, tokens_in=?, tokens_out=?, cost_usd_est=? WHERE id=?",
              (status, db.now(), tot[0], tot[1], round(tot[2], 6), r.id))
    log.info("run ended", extra={"data": {"status": status, "failed_steps": failed}})
    log.removeHandler(r._handler)
    r._handler.close()
    _run.set(getattr(r, "prev", None))
    _step.set(getattr(r, "prev_step", None))
    return status


@contextlib.contextmanager
def run(conn, kind, **kw):
    r = start_run(conn, kind, **kw)
    try:
        yield r
    except BaseException:
        end_run("failed")
        raise
    end_run()


def current_run():
    return _run.get()


# ---------------------------------------------------------------- steps

class Step:
    def __init__(self, sid, stage, object_id, parent):
        self.id, self.stage, self.object_id, self.parent = sid, stage, object_id, parent
        self.outputs = {}

    def set(self, **outputs):
        self.outputs.update(outputs)

    def skip(self, reason):
        self.outputs["skip_reason"] = reason
        raise SkipStep(reason)


@contextlib.contextmanager
def step(stage, object_id=None, sighting_id=None, inputs=None):
    r = _run.get()
    if r is None:
        raise RuntimeError("track.step() outside track.run()")
    parent = _step.get()
    object_id = object_id or (parent.object_id if parent else None)
    t0, started = time.perf_counter(), db.now()
    sid = db._insert(r.conn, "steps", dict(
        run_id=r.id, parent_step_id=parent.id if parent else None, object_id=object_id, sighting_id=sighting_id,
        stage=stage, status="running", started_at=started, inputs=redact(inputs or {})))
    s = Step(sid, stage, object_id, parent)
    token = _step.set(s)
    log.info(f"{stage} start", extra={"data": {"inputs": inputs}} if inputs else {})
    status, err = "ok", (None, None, None)
    try:
        yield s
    except (SkipStep, QuotaExceeded) as e:
        status = "skipped"
        s.outputs.setdefault("skip_reason", str(e))
    except BaseException as e:
        status = "failed"
        tb_path = r.path("tracebacks", f"step_{sid}.txt")
        with open(tb_path, "w", encoding="utf-8") as f:
            f.write(redact(traceback.format_exc()))
        err = (type(e).__name__, redact(str(e))[:2000], rel(tb_path))
        log.error(f"{stage} failed: {type(e).__name__}: {e}")
        raise
    finally:
        ms = round((time.perf_counter() - t0) * 1e3)
        r.conn.execute("UPDATE steps SET status=?, ended_at=?, duration_ms=?, outputs=?, error_type=?, error_msg=?, "
                       "traceback_path=?, object_id=? WHERE id=?",
                       (status, db.now(), ms, db.enc(redact(s.outputs)), *err, s.object_id, sid))
        if status != "failed":
            log.info(f"{stage} {status} ({ms} ms)", extra={"data": {"outputs": s.outputs}} if s.outputs else {})
        _step.reset(token)


def current_step():
    return _step.get()


def _sid():
    s = _step.get()
    return s.id if s else None


# ---------------------------------------------------------------- calls

def _conn():
    r = _run.get()
    if r is None:
        raise RuntimeError("tracking call outside track.run()")
    return r.conn


def cost_usd(model, tokens_in, tokens_out):
    p = config.load().get("prices", {}).get(model)
    return 0.0 if not p else (tokens_in * p[0] + tokens_out * p[1]) / 1e6


def record_llm_call(provider, model, tokens_in=0, tokens_out=0, latency_ms=None, stop_reason=None,
                    request_id=None, effort=None, cache_read=0, cache_write=0, quota_kind="request"):
    c = _conn()
    cost = cost_usd(model, tokens_in or 0, tokens_out or 0)
    lid = db._insert(c, "llm_calls", dict(
        step_id=_sid(), provider=provider, model=model, request_id=request_id, stop_reason=stop_reason,
        effort=effort, tokens_in=tokens_in, tokens_out=tokens_out, cache_read=cache_read, cache_write=cache_write,
        cost_usd_est=cost, latency_ms=latency_ms, created_at=db.now()))
    bump_quota(c, provider, model, quota_kind, 1, tokens_in or 0, tokens_out or 0)
    log.debug(f"llm {provider}/{model} in={tokens_in} out={tokens_out} stop={stop_reason} {latency_ms}ms")
    return lid


def record_tool_call(tool, args, result_summary=None, is_error=False, latency_ms=None, llm_call_id=None):
    tid = db._insert(_conn(), "tool_calls", dict(
        step_id=_sid(), llm_call_id=llm_call_id, tool=tool, args=redact(args),
        result_summary=redact(str(result_summary))[:2000] if result_summary is not None else None,
        is_error=int(bool(is_error)), latency_ms=latency_ms, created_at=db.now()))
    log.info(f"tool {tool}{' ERROR' if is_error else ''} ({latency_ms} ms)", extra={"data": {"args": args}})
    return tid


def record_http(method, url, status=None, nbytes=None, sha256=None, latency_ms=None, cached=False):
    return db._insert(_conn(), "http_requests", dict(
        step_id=_sid(), method=method, url=redact(url), status=status, bytes=nbytes, sha256=sha256,
        latency_ms=latency_ms, cached=int(bool(cached)), created_at=db.now()))


# ---------------------------------------------------------------- quota

def today():
    return dt.date.today().isoformat()


def bump_quota(conn, provider, model, kind, n=1, tokens_in=0, tokens_out=0):
    conn.execute("INSERT INTO quota_usage (day, provider, model, kind, count, tokens_in, tokens_out) "
                 "VALUES (?,?,?,?,?,?,?) ON CONFLICT(day, provider, model, kind) DO UPDATE SET "
                 "count=count+excluded.count, tokens_in=tokens_in+excluded.tokens_in, "
                 "tokens_out=tokens_out+excluded.tokens_out",
                 (today(), provider, model, kind, n, tokens_in, tokens_out))


def quota_used(conn, kind, model=None, day=None):
    sql, args = "SELECT COALESCE(SUM(count),0) FROM quota_usage WHERE day=? AND kind=?", [day or today(), kind]
    if model:
        sql += " AND model=?"
        args.append(model)
    return conn.execute(sql, args).fetchone()[0]


def check_quota(conn, kind, limit, model=None):
    """Raise QuotaExceeded when today's count for kind (optionally per model) has reached limit."""
    used = quota_used(conn, kind, model)
    if limit is not None and used >= limit:
        raise QuotaExceeded(f"daily {kind} quota reached ({used}/{limit}{' ' + model if model else ''})")
    return used


def quota_status(conn):
    q = config.load().get("quota", {})
    month = today()[:7]
    out = {
        "day": today(),
        "grounded_search": {"used": quota_used(conn, "grounded_search"),
                            "limit": q.get("grounded_search_per_day")},
        "vision_web_detection_month": {
            "used": conn.execute("SELECT COALESCE(SUM(count),0) FROM quota_usage WHERE kind='vision_web_detection' "
                                 "AND substr(day,1,7)=?", (month,)).fetchone()[0],
            "limit": 1000},
        "requests": [],
    }
    limits = q.get("requests_per_day", {})
    for r in conn.execute("SELECT provider, model, SUM(count) AS n, SUM(tokens_in) AS ti, SUM(tokens_out) AS tout "
                          "FROM quota_usage WHERE day=? AND kind='request' GROUP BY provider, model", (today(),)):
        out["requests"].append({"provider": r["provider"], "model": r["model"], "used": r["n"],
                                "limit": limits.get(r["model"]), "tokens_in": r["ti"], "tokens_out": r["tout"]})
    for m, lim in limits.items():
        if not any(x["model"] == m for x in out["requests"]):
            out["requests"].append({"provider": "gemini", "model": m, "used": 0, "limit": lim,
                                    "tokens_in": 0, "tokens_out": 0})
    return out
