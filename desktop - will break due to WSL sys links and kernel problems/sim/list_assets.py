"""List files in an Isaac assets folder (e.g. which YCB props exist).

Run:  .\\run_isaac.ps1 sim\\list_assets.py Isaac/Props/YCB/Axis_Aligned_Physics
"""
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.client
from common import log, shutdown
from isaacsim.storage.native import get_assets_root_path


def main() -> int:
    for folder in sys.argv[1:]:
        url = f"{get_assets_root_path()}/{folder}"
        result, entries = omni.client.list(url)
        log(f"[assets] {folder}: {result}")
        for e in sorted(entries, key=lambda e: e.relative_path):
            log(f"   {e.relative_path}")
    return 0


if __name__ == "__main__":
    shutdown(app, main())
