"""Check that the Isaac Sim assets this project uses can be resolved.

What this teaches: Isaac Sim assets (robots, props, YCB objects) live on a remote/Nucleus
"assets root"; `get_assets_root_path()` returns it, and every USD path in config/ is relative to it.

Run (Windows):  C:\\isaac-sim\\python.bat sim\\check_assets.py
"""

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})  # must be created before importing other isaacsim/omni modules

import omni.client
from common import load_yaml, log, shutdown
from pnp_core.catalog import load_robot  # noqa: E402
from isaacsim.storage.native import get_assets_root_path


def exists(url: str) -> bool:
    result, _ = omni.client.stat(url)
    return result == omni.client.Result.OK


def main() -> int:
    root = get_assets_root_path()
    log(f"[assets] root = {root}")
    if root is None:
        log("[assets] ERROR: could not resolve the assets root (offline? see docs/learning/00_setup.md)")
        return 1

    paths = {"robot": load_robot()["sim"]["usd"]}
    paths.update({name: o["usd"] for name, o in load_yaml("config/objects.yaml").items()})

    ok = True
    for name, rel in paths.items():
        found = exists(f"{root}/{rel}")
        ok &= found
        log(f"[assets] {'OK     ' if found else 'MISSING'} {name:18s} {rel}")
    return 0 if ok else 1


if __name__ == "__main__":
    shutdown(app, main())
