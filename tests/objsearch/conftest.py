"""objsearch test tiers (see scripts/test_objsearch.sh):
unit (default, fakes only) | bpy (real Blender) | replay (recorded cassettes) | live (real APIs, opt-in) | bench."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
SEED = os.path.join(ROOT, "catalog", "seed.yaml")
TIERS = ("bpy", "replay", "live", "bench")


def pytest_addoption(parser):
    parser.addoption("--max-requests", type=int, default=None, help="live tier: cap on real API requests")
    parser.addoption("--budget-usd", type=float, default=None, help="live tier: cap on paid API spend")


def pytest_configure(config):
    for t in TIERS:
        config.addinivalue_line("markers", f"{t}: objsearch {t} tier (deselected by default)")


def pytest_collection_modifyitems(config, items):
    expr = config.getoption("-m") or ""
    for item in items:
        for t in TIERS:
            if t in item.keywords and t not in expr:
                item.add_marker(pytest.mark.skip(reason=f"{t} tier: run with -m {t}"))
        if "live" in item.keywords and config.getoption("--max-requests") is None:
            item.add_marker(pytest.mark.skip(reason="live tier needs --max-requests (and --budget-usd for paid APIs)"))


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "objsearch_data"
    monkeypatch.setenv("OBJSEARCH_DATA", str(d))
    monkeypatch.setenv("OBJSEARCH_AUTO_ACQUIRE", "0")  # unit tests never start real model builds
    monkeypatch.setenv("OBJSEARCH_SIM", "0")            # ...nor Blender for the sim.usda layer
    monkeypatch.setenv("OBJSEARCH_AUTO_DOCS", "0")      # ...nor datasheet research
    monkeypatch.setenv("OBJSEARCH_NO_GITHUB", "1")      # ...nor real GitHub searches (fakes patch github.available)
    return d


@pytest.fixture
def conn(data_dir):
    from objsearch import db
    c = db.connect()
    yield c
    c.close()


@pytest.fixture
def seeded(conn):
    from objsearch import db
    db.import_seed(conn, SEED)
    return conn
