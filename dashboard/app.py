"""Local dashboard for dino-box. Runs on this PC and pulls everything over SSH.

SSH port forwarding is disabled on the box, so nothing is tunnelled: each API call runs a
short `ssh dino-box <cmd>` and relays the result.

    .venv/Scripts/python -m uvicorn dashboard.app:app --port 8765     ->  http://127.0.0.1:8765
"""
import json
import os
import re
import subprocess
import threading
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response

HOST = os.environ.get("DINO_HOST", "dino-box")
CODE = "/workspace/dino/code"
RUNS = "/workspace/dino/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
SAFE_NAME = re.compile(r"^[\w.-]+$")
SAFE_REL = re.compile(r"^[\w.-]+(/[\w.-]+)*$")
MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".json": "application/json",
         ".csv": "text/plain", ".txt": "text/plain", ".md": "text/plain", ".yaml": "text/plain", ".log": "text/plain"}

app = FastAPI(title="dino-box dashboard")


class Conn:
    """One persistent SSH transport (each handshake to the box costs ~3 s); a channel per command.

    Uses the host's entry in ~/.ssh/config. Falls back to the ssh CLI if paramiko is unavailable.
    """

    def __init__(self, host):
        self.host, self.client, self.lock = host, None, threading.Lock()

    def _connect(self):
        import paramiko

        cfg = paramiko.SSHConfig.from_path(os.path.expanduser("~/.ssh/config")).lookup(self.host)
        c = paramiko.SSHClient()
        c.load_host_keys(os.path.expanduser("~/.ssh/known_hosts")) if os.path.exists(os.path.expanduser("~/.ssh/known_hosts")) else None
        c.set_missing_host_key_policy(paramiko.RejectPolicy())  # host must already be trusted by `ssh dino-box`
        c.connect(cfg["hostname"], port=int(cfg.get("port", 22)), username=cfg.get("user"),
                  key_filename=[os.path.expanduser(k) for k in cfg.get("identityfile", [])] or None,
                  timeout=15, banner_timeout=15, auth_timeout=15, look_for_keys=False, allow_agent=False)
        c.get_transport().set_keepalive(30)
        return c

    def client_(self):
        with self.lock:
            t = self.client.get_transport() if self.client else None
            if not (t and t.is_active()):
                self.client = self._connect()
            return self.client

    def run(self, cmd, timeout):
        for attempt in (0, 1):  # one reconnect on a dropped transport
            try:
                ch = self.client_().get_transport().open_session(timeout=15)
                break
            except Exception:
                with self.lock:
                    self.client = None
                if attempt:
                    raise
        ch.settimeout(timeout)
        ch.exec_command(cmd)
        out = ch.makefile("rb").read()  # blocks until EOF, so output is never truncated
        err = ch.makefile_stderr("rb").read()
        code = ch.recv_exit_status()
        ch.close()
        return code, out, err


CONN = Conn(HOST)


def _run_cli(cmd, timeout):
    # -n / DEVNULL: an inherited stdin pipe makes Windows OpenSSH hang after the remote command exits
    r = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, cmd],
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def ssh(cmd, timeout=30, binary=False):
    try:
        try:
            code, out, err = CONN.run(cmd, timeout)
        except ImportError:
            code, out, err = _run_cli(cmd, timeout)
    except Exception as e:  # paramiko.SSHException, socket timeouts, CLI timeout
        raise HTTPException(504, f"ssh failed ({type(e).__name__}: {e}): {cmd[:80]}")
    if code != 0:
        raise HTTPException(502, err.decode(errors="replace")[-800:] or f"exit {code}")
    return out if binary else out.decode(errors="replace")


def _rel(path):
    if not SAFE_REL.match(path) or ".." in path.split("/"):
        raise HTTPException(400, "bad path")
    return f"{RUNS}/{path}"


_cache = {"t": 0.0, "data": None}
_lock = threading.Lock()


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "index.html"))


@app.get("/api/status")
def status():
    with _lock:  # many browser tabs share one ssh call per 2s
        if time.time() - _cache["t"] > 2:
            t = time.perf_counter()
            data = json.loads(ssh(f"python3 {CODE}/remote/status.py"))
            data["ssh_ms"] = round((time.perf_counter() - t) * 1e3)
            _cache.update(t=time.time(), data=data)
        return _cache["data"]


@app.get("/api/run/{name}")
def run(name: str):
    if not SAFE_NAME.match(name):
        raise HTTPException(400, "bad name")
    out = ssh(f"cd {RUNS}/{name} && cat metrics.json 2>/dev/null; echo '<<<SPLIT>>>'; "
              f"find . -maxdepth 2 -type d | sort | head -50")
    metrics, dirs = out.split("<<<SPLIT>>>")
    return {"metrics": json.loads(metrics) if metrics.strip() else None,
            "dirs": [d[2:] for d in dirs.split() if d not in (".", "")]}


@app.get("/api/log")
def log(path: str, lines: int = Query(200, le=5000), head: bool = False):
    return Response(ssh(f"{'head' if head else 'tail'} -n {int(lines)} {_rel(path)}"), media_type="text/plain")


@app.get("/api/ls")
def ls(path: str, limit: int = Query(200, le=2000)):
    out = ssh(f"cd {_rel(path)} && ls -1p | head -n {int(limit)}")
    return {"path": path, "entries": out.split()}


@app.get("/api/file")
def file(path: str):
    full = _rel(path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in MEDIA:
        raise HTTPException(400, "unsupported file type")
    return Response(ssh(f"cat {full}", binary=True), media_type=MEDIA[ext],
                    headers={"Cache-Control": "max-age=30"})


@app.get("/api/remote-ping")
def remote_ping():
    """Round trip: this PC -> ssh -> curl on the box -> FastAPI ping app on 127.0.0.1:8000."""
    t = time.perf_counter()
    out = ssh("curl -s -m 5 127.0.0.1:8000/ping; echo; curl -s -m 20 127.0.0.1:8000/gpu", timeout=40)
    ms = round((time.perf_counter() - t) * 1e3)
    parts = [p for p in out.splitlines() if p.strip()]
    try:
        return {"ok": True, "roundtrip_ms": ms, "ping": json.loads(parts[0]), "gpu": json.loads(parts[1])}
    except (IndexError, ValueError):
        return {"ok": False, "roundtrip_ms": ms, "error": "ping app not responding on the box",
                "hint": "bash scripts/bg.sh ping uvicorn remote.ping_app:app --host 127.0.0.1 --port 8000"}
