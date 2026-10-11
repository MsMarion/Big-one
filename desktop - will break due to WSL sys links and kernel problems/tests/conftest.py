import sys
from pathlib import Path

# Make `pnp_core` and `tests.synthetic` importable without installing anything.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
