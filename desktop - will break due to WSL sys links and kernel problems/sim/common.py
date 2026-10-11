"""Small helpers shared by every standalone Isaac Sim script in sim/.

Gotcha this file exists for: `SimulationApp.close()` tears the process down, so
  - anything still sitting in Python's stdout buffer is lost (prints "disappear"), and
  - a later `sys.exit(code)` never runs (exit code is always 0).
`log()` flushes immediately and `shutdown()` closes the app with a real exit code.
Import this module only *after* `SimulationApp(...)` has been created.
"""

import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))         # pnp_core importable from every sim script
CONFIG = REPO / "config"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_yaml(path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = REPO / path
    return yaml.safe_load(path.read_text())


def shutdown(app, code: int = 0) -> None:
    """Close the Kit app and exit with `code` (close() alone would always exit 0)."""
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        app.close()
    finally:
        os._exit(code)
