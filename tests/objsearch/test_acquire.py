import pytest

from objsearch import acquire, db, models, track
from objsearch.llm import gemini_backend


@pytest.fixture
def fake_build(monkeypatch):
    """Record what acquire tries to build; optionally fail some sources."""
    built, fail = [], set()

    def add_model(conn, oid, src, license=None, source_url=None, note=None, **kw):
        if any(f in str(src) for f in fail):
            raise RuntimeError("conversion failed")
        built.append((oid, str(src), license, note))
        with track.run(conn, "model"):
            aid = db.add_asset(conn, oid, "glb", f"model_cache/x/{oid}.glb")
        db.add_candidate(conn, oid, source_url=source_url or str(src), license=license, rank=1, chosen=1,
                         scores={"flags": []})
        return {"run_id": "r", "dims_mm": [1, 2, 3], "flags": [], "asset": aid}
    monkeypatch.setattr(models, "add_model", add_model)
    return built, fail


def test_name_matching_is_strict():
    cands = [("a", "RaspberryPi_Pico"), ("b", "RaspberryPi_Pico_H"), ("c", "RaspberryPi_Pico_W"), ("d", "lego_brick"),
             ("e", "TO-5-2"), ("f", "ESP-12E")]
    assert acquire.match_names(["Raspberry Pi Pico H (with headers)"], cands) == "b"   # most specific wins
    assert acquire.match_names(["Raspberry Pi Pico"], cands) == "a"                     # W/H variants not covered
    assert acquire.match_names(["LEGO Brick 2 x 4"], cands) is None                     # no distinctive token
    assert acquire.match_names(["plug to 5.5 x 2 adapter"], cands) is None
    assert acquire.match_names(["ESP-12E"], cands) == "f"


def test_ladder_known_source_then_learns(seeded, fake_build):
    built, _ = fake_build
    db.update_object(seeded, "esp-01", {"model_hints": [{"url": "https://maker.test/esp01.step", "license": "CC0"}]})
    out = acquire.acquire(seeded, "esp-01", use_agent=False)
    assert out["how"] == "known source" and built[0][1] == "https://maker.test/esp01.step"
    assert acquire.acquire(seeded, "esp-01", use_agent=False)["how"] == "already has a model"


def test_ladder_falls_through_to_placeholder_and_skips_agent_without_key(seeded, fake_build, monkeypatch):
    built, _ = fake_build
    monkeypatch.setattr(gemini_backend, "api_key", lambda: None)
    _no_fast_sources(monkeypatch)
    out = acquire.acquire(seeded, "lego-3001")
    assert out["how"] == "placeholder" and built[0][1].endswith("lego-3001.glb") and built[0][2] == "generated"
    steps = {s["stage"]: (s["status"], (s["outputs"] or {}).get("skip_reason")) for s in db.get_run(seeded, out["run_id"])["steps"]}
    assert steps["source_agent"][0] == "skipped" and "no LLM key" in steps["source_agent"][1]
    flags = db.get_object_full(seeded, "lego-3001")["candidates"][0]["scores"]["flags"]
    assert flags and flags[0].startswith("placeholder model")


def test_bad_candidate_moves_on_and_library_hit_is_learned(seeded, fake_build, monkeypatch):
    built, fail = fake_build
    fail.add("broken.step")
    monkeypatch.setattr(acquire, "_known", lambda o: {"src": "https://x.test/broken.step", "license": "?", "how": "known source"})
    monkeypatch.setattr(acquire, "_kicad", lambda o: {"src": "https://lib.test/ESP-01.step", "license": "CC-BY-SA",
                                                      "source_url": "https://lib.test/page", "how": "KiCad library"})
    out = acquire.acquire(seeded, "esp-01", use_agent=False)
    assert out["how"] == "KiCad library" and out["tried"] == ["known source", "KiCad library"]
    hints = db.get_object(seeded, "esp-01")["model_hints"]
    assert hints[0]["url"] == "https://lib.test/ESP-01.step" and hints[0]["verified"] is True


def _no_fast_sources(monkeypatch):
    for name in ("_ldraw", "_github_boards", "_kicad", "_open_lib", "_github", "_boardgen"):
        monkeypatch.setattr(acquire, name, lambda o: None)


def test_agent_stops_at_first_good_fit_and_replaces_placeholder(seeded, fake_build, monkeypatch):
    from objsearch.llm.fake_backend import FakeLLM
    from objsearch.sources import github
    _no_fast_sources(monkeypatch)
    monkeypatch.setattr(github, "available", lambda: True)
    monkeypatch.setattr(github, "search_cad", lambda q, limit=8, exts=None: [
        {"name": "hc.step", "repo": "a/b", "url": "https://maker.test/hc.step", "page": "https://maker.test/hc",
         "license": "CC-BY-4.0"}])
    tried = []
    monkeypatch.setattr(acquire, "try_model", lambda url, o: tried.append(url) or {"ok": True, "verdict": "good fit"})
    agent = FakeLLM([{"tool_calls": [{"name": "github_cad_search", "args": {"query": "HC-SR04"}}]},
                     {"tool_calls": [{"name": "try_model", "args": {"url": "https://maker.test/hc.step"}}]},
                     {"tool_calls": [{"name": "web_search", "args": {"query": "never reached"}}]}])
    out = acquire.acquire(seeded, "hc-sr04", backend=agent)
    assert out["how"] == "research agent" and tried == ["https://maker.test/hc.step"]
    built = fake_build[0]
    assert built[0][2] == "generated"  # placeholder first: something visible within seconds
    assert built[1][1] == "https://maker.test/hc.step" and built[1][2] == "CC-BY-4.0"  # license kept from search
    assert out["placeholder_first"] is True
    tools_used = [r[0] for r in seeded.execute("SELECT tool FROM tool_calls ORDER BY id")]
    assert tools_used == ["github_cad_search", "try_model"]  # stopped right after the good fit
    # the researched model replaced the placeholder: one model asset, one chosen candidate, no placeholder flag
    assert seeded.execute("SELECT COUNT(*) FROM assets WHERE object_id='hc-sr04'").fetchone()[0] == 1
    chosen = [c for c in db.get_object_full(seeded, "hc-sr04")["candidates"] if c["chosen"]]
    assert len(chosen) == 1 and chosen[0]["license"] == "CC-BY-4.0" and not chosen[0]["scores"]["flags"]


def test_github_rung_tests_best_named_files_without_llm(seeded, fake_build, monkeypatch):
    from objsearch.sources import github
    for n in ("_kicad", "_open_lib", "_github_boards", "_boardgen"):
        monkeypatch.setattr(acquire, n, lambda o: None)
    monkeypatch.setattr(github, "available", lambda: True)
    hit = lambda n: {"name": n, "repo": "a/b", "url": f"https://raw.test/{n}", "page": f"https://gh.test/{n}",
                     "license": "MIT"}
    monkeypatch.setattr(github, "search_cad", lambda q, limit=8, exts=None: [
        hit("enclosure.step"), hit("Display_OLED_SSD1306_1.3inch.step"), hit("OLED_SSD1306_0.96in_I2C.step")])
    tried = []

    def try_model(url, o):
        tried.append(url)
        return {"ok": True, "verdict": "good fit" if "0.96" in url else "size mismatch"}
    monkeypatch.setattr(acquire, "try_model", try_model)
    out = acquire.acquire(seeded, "ssd1306-oled-096", backend=object())
    assert out["how"] == "GitHub CAD search" and out["placeholder_first"] is False
    assert tried == ["https://raw.test/OLED_SSD1306_0.96in_I2C.step"]  # best-named file first; no-name file skipped
    assert fake_build[0][0][1:3] == ("https://raw.test/OLED_SSD1306_0.96in_I2C.step", "MIT")
    hints = db.get_object(seeded, "ssd1306-oled-096")["model_hints"]
    assert hints[0]["url"] == "https://raw.test/OLED_SSD1306_0.96in_I2C.step"  # learned: next time it's instant


def test_failed_research_build_keeps_placeholder(seeded, fake_build, monkeypatch):
    built, fail = fake_build
    fail.add("broken.step")
    _no_fast_sources(monkeypatch)
    monkeypatch.setattr(acquire, "_agent", lambda conn, o, b: {"src": "https://x.test/broken.step", "license": "?",
                                                                "how": "research agent"})
    out = acquire.acquire(seeded, "hc-sr04", backend=object())
    assert out["how"] == "placeholder" and out["tried"] == ["placeholder", "research agent"]
    assert seeded.execute("SELECT COUNT(*) FROM assets WHERE object_id='hc-sr04'").fetchone()[0] == 1


def test_github_cad_search_returns_direct_links(conn, monkeypatch):
    import json
    from objsearch.sources import github
    github._license_raw.cache_clear()
    calls = []

    def gh(path, params=None):
        calls.append(path)
        if path == "search/code":
            items = [] if params["q"].endswith("extension:stp") else [
                {"path": "3d/oled 0.96.step", "html_url": "https://github.com/a/b/blob/main/3d/oled%200.96.step",
                 "repository": {"full_name": "a/b"}}]
            body = {"items": items}
        else:
            body = {"license": {"spdx_id": "MIT"}}
        return f"github:{path}", True, json.dumps(body), 10, 5
    monkeypatch.setattr(github, "_gh", gh)
    with track.run(conn, "test"):
        res = github.search_cad("SSD1306")
    assert res == [{"name": "oled 0.96.step", "repo": "a/b", "license": "MIT",
                    "url": "https://raw.githubusercontent.com/a/b/main/3d/oled%200.96.step",
                    "page": "https://github.com/a/b/blob/main/3d/oled%200.96.step"}]
    assert calls.count("repos/a/b") == 1
    assert conn.execute("SELECT COUNT(*) FROM http_requests WHERE url LIKE 'github:%'").fetchone()[0] == 3


def test_size_fit_uses_footprint_and_tolerates_height():
    oled = [27.3, 27.8, 3.7]
    fit = acquire.size_fit([0.0248, 0.0269, 0.0069], oled)  # real 0.96" model in metres, headers make it taller
    assert fit["verdict"] == "good fit" and fit["units_guess"] == "m" and "height_note" in fit
    assert acquire.size_fit([3.0, 35.1, 35.6], oled)["verdict"] == "size mismatch"  # the 1.3" Adafruit board
    assert acquire.size_fit([27.5, 27.5, 30.0], oled)["verdict"] == "size mismatch"  # far too tall


def test_cache_hit_for_plain_mesh(seeded, data_dir, tmp_path, monkeypatch):
    """A cached STEP/mesh build (Blender's "parts" is a mesh count, not a board report) is reused, not rebuilt."""
    import json
    import os
    from objsearch import config
    src = tmp_path / "part.stl"
    src.write_text("solid x\nendsolid x\n")
    monkeypatch.setattr(models, "_build", lambda *a, **k: pytest.fail("should have used the cache"))
    import hashlib
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    o = db.get_object(seeded, "esp-01")
    from objsearch.kicad import board_colors
    key = models.build_key(seeded, sha, board_colors(o.get("colors")), o["dims_mm"], str(src))
    out = os.path.join(config.data_dir(), "model_cache", key)
    os.makedirs(out)
    for f in ("model.glb", "asset.usd", "thumb.png", "source.stl"):
        open(os.path.join(out, f), "wb").close()
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump({"files": {"glb": "model.glb", "usd": "asset.usd", "thumb": "thumb.png", "source": "source.stl"},
                   "dims_mm": [1, 2, 3],
                   "parts": 7, "flags": [], "tri_count": 2000}, f)
    res = models.add_model(seeded, "esp-01", str(src), license="CC0")
    assert res["dims_mm"] == [1, 2, 3]


def test_tiers_detail_classification():
    assert acquire.detail({"triangles": 50}) == "box"
    assert acquire.detail({"triangles": 748, "parts": 7}) == "simple"
    assert acquire.detail({"triangles": 748, "parts": 14}) == "detailed"
    assert acquire.detail({"triangles": 25000, "parts": 2}) == "detailed"
    assert acquire.tier_of({"detail": "board"}) == 1 and acquire.tier_of({"triangles": 748, "parts": 7}) == 4


def _ladder(monkeypatch, **cands):
    for n in ("_known", "_ldraw", "_github_boards", "_kicad", "_open_lib", "_github", "_boardgen"):
        monkeypatch.setattr(acquire, n, (lambda c: (lambda o: c))(cands.get(n)))


def test_simple_step_is_only_a_fallback_and_replica_wins(seeded, fake_build, monkeypatch):
    built, _ = fake_build
    simple = {"src": "https://x.test/simple.step", "license": "MIT", "how": "GitHub CAD search", "tier": 4}
    replica = {"src": "/tmp/board.kicad_pcb", "license": "rebuilt", "how": "KiCad replica", "tier": 3}
    _ladder(monkeypatch, _github=simple, _boardgen=replica)
    out = acquire.acquire(seeded, "hc-sr04", use_agent=False)
    assert out["how"] == "KiCad replica" and out["tier"] == 3 and [b[1] for b in built] == ["/tmp/board.kicad_pcb"]


def test_simple_step_beats_photo_box(seeded, fake_build, monkeypatch):
    built, _ = fake_build
    _ladder(monkeypatch, _github={"src": "https://x.test/s.step", "license": "MIT", "how": "GitHub CAD search",
                                  "tier": 4})
    out = acquire.acquire(seeded, "hc-sr04", use_agent=False)
    assert out["tier"] == 4 and built[0][1] == "https://x.test/s.step"
    flags = [c for c in db.get_object_full(seeded, "hc-sr04")["candidates"] if c["chosen"]][0]["scores"]["flags"]
    assert any(f.startswith("simple model") for f in flags)


def test_exact_board_short_circuits(seeded, fake_build, monkeypatch):
    built, _ = fake_build
    board = {"src": "https://x.test/hc.kicad_pcb", "license": "CC0", "how": "GitHub board design", "tier": 1}
    _ladder(monkeypatch, _github_boards=board, _boardgen={"src": "x", "how": "KiCad replica", "tier": 3})
    out = acquire.acquire(seeded, "hc-sr04")
    assert out["tier"] == 1 and len(built) == 1
    assert db.get_object(seeded, "hc-sr04")["model_hints"][0]["tier"] == 1  # learned with its tier


def test_agent_only_replaces_with_a_better_tier(seeded, fake_build, monkeypatch):
    built, _ = fake_build
    _ladder(monkeypatch, _boardgen={"src": "/tmp/b.kicad_pcb", "how": "KiCad replica", "tier": 3})
    monkeypatch.setattr(acquire, "_agent", lambda c, o, b: {"src": "https://x.test/s.step", "how": "research agent",
                                                            "tier": 4})
    out = acquire.acquire(seeded, "hc-sr04", backend=object())
    assert out["tier"] == 3 and len(built) == 1  # a simple STEP from the agent never replaces the replica
    monkeypatch.setattr(acquire, "_agent", lambda c, o, b: {"src": "https://x.test/d.step", "how": "research agent",
                                                            "tier": 2})
    out = acquire.acquire(seeded, "hc-sr04", backend=object(), force=True)
    assert out["tier"] == 2 and built[-1][1] == "https://x.test/d.step"


def test_broken_source_does_not_end_the_ladder(seeded, fake_build, monkeypatch):
    def boom(o):
        raise TypeError("bug in a source")
    _ladder(monkeypatch, _github={"src": "https://x.test/s.step", "how": "GitHub CAD search", "tier": 4})
    monkeypatch.setattr(acquire, "_kicad", boom)
    assert acquire.acquire(seeded, "hc-sr04", use_agent=False)["tier"] == 4


def test_github_query_cleanup_and_rate_limit_retry(monkeypatch):
    import json
    from objsearch.sources import github
    assert github.clean_query('2.0" IPS TFT display GMT020-02-8P (ST7789V, 240x320, SPI)') == \
        "2.0 IPS TFT display GMT020-02-8P ST7789V"
    assert github.clean_query("repo:x/y extension:step") == "repo x/y extension step"
    github._CACHE.clear()
    calls, waits = [], []
    def gh(path, params=None):
        calls.append(path)
        if path == "search/code" and calls.count("search/code") == 1:
            return "github:search/code", False, "gh: API rate limit exceeded for user (HTTP 403)", 0, 3
        return f"github:{path}", True, json.dumps({"items": []}), 10, 3
    monkeypatch.setattr(github, "_gh", gh)
    monkeypatch.setattr(github, "_wait_for_reset", lambda max_wait=65: waits.append(1))
    monkeypatch.setattr(github.track, "record_http", lambda *a, **k: None)
    assert github._api("search/code", {"q": "x"}) == {"items": []} and waits == [1]
    assert github._api("search/code", {"q": "x"}) == {"items": []} and calls.count("search/code") == 2  # cached
