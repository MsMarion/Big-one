import io
import os
import time

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from google.genai import errors as gerrors
from google.genai import types as gt
from PIL import Image

from objsearch import config, db, identify, track
from objsearch.llm import gemini_backend, tools
from objsearch.llm.base import Budget, Image as LLMImage
from objsearch.llm.fake_backend import FakeLLM
from objsearch.sources import brickognize, web


def _jpeg(color=(40, 70, 160)):
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color).save(buf, "JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------- gemini backend with a fake client

def _usage(i=100, o=20):
    return gt.GenerateContentResponseUsageMetadata(prompt_token_count=i, candidates_token_count=o)


def _call(name, args):
    return gt.GenerateContentResponse(candidates=[gt.Candidate(
        content=gt.Content(role="model", parts=[gt.Part(function_call=gt.FunctionCall(name=name, args=args))]),
        finish_reason=gt.FinishReason.STOP)], usage_metadata=_usage())


def _text(t, grounding=None):
    return gt.GenerateContentResponse(candidates=[gt.Candidate(
        content=gt.Content(role="model", parts=[gt.Part(text=t)]), finish_reason=gt.FinishReason.STOP,
        grounding_metadata=grounding)], usage_metadata=_usage())


class FakeClient:
    def __init__(self, by_model):
        self.queues, self.calls = by_model, []
        self.models = self

    def generate_content(self, model, contents, config=None):
        self.calls.append((model, contents, config))
        nxt = self.queues[model].pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


FINAL = {"name": "ESP-01 (ESP8266) WiFi module", "category": "module", "confidence": 0.9,
         "evidence_urls": ["https://example.test/esp01"], "catalog_id": "esp-01", "reasoning": "reads ESP-01"}


def test_gemini_loop(seeded, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    grounding = gt.GroundingMetadata(web_search_queries=["esp-01 module"], grounding_chunks=[
        gt.GroundingChunk(web=gt.GroundingChunkWeb(uri="https://example.test/esp01", title="example.test"))])
    client = FakeClient({
        "agent-m": [_call("catalog_lookup", {"text": "ESP-01"}), _call("web_search", {"query": "ESP-01 module"}),
                    _call("submit_result", {"name": "x"}),  # invalid -> error back to the model
                    _text("I think it's an ESP-01."),       # text instead of submit -> one nudge
                    _call("submit_result", FINAL)],
        "search-m": [gerrors.ClientError(429, {"error": {"message": "slow down"}}), _text("ESP-01 is 24.8 x 14.3 mm", grounding)],
    })
    be = gemini_backend.GeminiBackend("agent-m", "search-m", client=client)
    handlers = {"catalog_lookup": tools.catalog_lookup_handler(seeded), "web_search": lambda a: be.grounded_search(a["query"])}
    with track.run(seeded, "test") as r:
        with track.step("identify"):
            res = be.run("sys", ["hello", LLMImage(data=_jpeg(), label="photo_1")],
                         tools.build_tools(handlers, tools.IDENTIFY_TOOLS), tools.IDENTIFICATION_SCHEMA, Budget())
    assert res.stop_reason == "final" and res.output["catalog_id"] == "esp-01"
    run = db.get_run(seeded, r.id)
    tc = [(t["tool"], bool(t["is_error"])) for t in run["tool_calls"]]
    assert tc == [("catalog_lookup", False), ("web_search", False), ("submit_result", True)]
    ws = next(t for t in run["tool_calls"] if t["tool"] == "web_search")
    assert "example.test/esp01" in ws["result_summary"] and "grounded" in ws["result_summary"]
    stops = [c["stop_reason"] for c in run["llm_calls"]]
    assert "error 429" in stops and len(run["llm_calls"]) == 7
    assert track.quota_used(seeded, "grounded_search") == 2  # the 429 attempt counts too
    # history sent back: model turns appended unchanged, tool replies as function responses, image inline
    last_contents = client.calls[-1][1]
    assert any(p.inline_data for p in last_contents[0].parts)
    assert any(p.function_response and p.function_response.name == "catalog_lookup"
               for c in last_contents for p in (c.parts or []))
    decl_names = [d.name for d in client.calls[0][2].tools[0].function_declarations]
    assert "submit_result" in decl_names and "web_search" in decl_names


def test_gemini_quota_and_empty(seeded, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    client = FakeClient({"agent-m": [gt.GenerateContentResponse(candidates=[], usage_metadata=_usage())]})
    be = gemini_backend.GeminiBackend("agent-m", "search-m", client=client)
    with track.run(seeded, "test"):
        with track.step("identify"):
            assert be.run("sys", ["x"], [], tools.IDENTIFICATION_SCHEMA, Budget()).stop_reason == "empty_response"
        track.bump_quota(seeded, "gemini", "search-m", "grounded_search", 500)
        from objsearch.sources import ddg
        monkeypatch.setattr(ddg, "search", lambda q, n=6: {"results": [], "trust": "fallback"})
        assert be.grounded_search("anything")["trust"] == "fallback"  # quota used up -> keyless fallback
        monkeypatch.setitem(config.load(), "search_mode", "grounded")
        be._grounding_off = None
        with pytest.raises(track.QuotaExceeded):
            be.grounded_search("anything")
        monkeypatch.setitem(config.load(), "search_mode", "auto")
    blocked = FakeClient({"agent-m": [gerrors.ClientError(429, {"error": {"message": "x"}})] * 3})
    with track.run(seeded, "test"):
        with pytest.raises(track.QuotaExceeded):
            gemini_backend.GeminiBackend("agent-m", "s", client=blocked)._generate("agent-m", "x", None)


# ---------------------------------------------------------------- identify pipeline

def _fake(final, calls=()):
    return FakeLLM([{"tool_calls": list(calls)}] * bool(calls) + [{"final": final}])


def test_identify_matches_catalog_then_confirm(seeded):
    res = identify.identify(seeded, [_jpeg(), _jpeg((200, 30, 30))], hint="blue board", dims_mm=[25, 14, 11],
                            backend=_fake(FINAL, [{"name": "catalog_lookup", "args": {"text": "ESP-01"}}]))
    assert res["match"] == "catalog" and res["object_id"] == "esp-01" and len(res["photos"]) == 2
    v = identify.run_view(seeded, res["run_id"])
    assert v["status"] == "ok" and [s["stage"] for s in v["steps"]] == ["intake", "identify", "record"]
    assert v["activity"][0]["tool"] == "catalog_lookup" and v["object"]["id"] == "esp-01"
    out = identify.confirm(seeded, res["run_id"], True, owned=True)
    o = db.get_object_full(seeded, "esp-01")
    assert out["action"] == "confirmed" and o["owned"] == 1 and o["origin"] == "given" and o["include_in_search"] == 1
    assert sum(p["source"] == "user" for p in o["photos"]) == 2
    with pytest.raises(ValueError):
        identify.confirm(seeded, res["run_id"], True)


def test_identify_new_item_and_correction(seeded):
    new = {**FINAL, "name": "Mystery sensor board XYZ-9", "catalog_id": "", "confidence": 0.4,
           "evidence_urls": ["https://example.test/xyz", "javascript:alert(1)"], "expected_dims_mm": [30, 20, 12]}
    res = identify.identify(seeded, [_jpeg()], backend=_fake(new))
    o = db.get_object_full(seeded, res["object_id"])
    assert res["match"] == "new" and o["origin"] == "found" and o["status"] == "draft" and o["dims_mm"] == [30, 20, 12]
    assert [e["url"] for e in o["evidence"]] == ["https://example.test/xyz"]  # non-http dropped
    out = identify.confirm(seeded, res["run_id"], False, owned=True, name="My custom sensor")
    assert out["action"] == "corrected"
    assert db.get_object(seeded, res["object_id"])["status"] == "rejected"
    mine = db.get_object_full(seeded, out["object_id"])
    assert mine["origin"] == "given" and mine["owned"] == 1 and mine["photos"]


def test_identify_survives_agent_crash(seeded):
    res = identify.identify(seeded, [_jpeg()], backend=FakeLLM([{"raise": "ConnectionError: 503"}]))
    assert res["match"] == "none" and "ConnectionError" in res["stop_reason"]
    steps = {s["stage"]: s["status"] for s in db.get_run(seeded, res["run_id"])["steps"]}
    assert steps == {"intake": "ok", "identify": "failed", "record": "ok"}


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(gemini_backend, "api_key", lambda: None)


def _brick_mock(m, score=0.82, part="3001"):
    m.post(brickognize.API).mock(return_value=httpx.Response(200, json={
        "bounding_box": {"score": 0.97}, "items": [{"id": part, "name": "Brick 2 x 4", "type": "part", "category": "Brick",
                                                    "score": score, "external_sites": [{"name": "bricklink", "url": "https://bricklink.test/3001"}]}]}))


def test_keyless_lego(seeded, no_key):
    with respx.mock() as m:
        _brick_mock(m)
        res = identify.identify(seeded, [_jpeg((200, 200, 0))])
        assert res["match"] == "catalog" and res["object_id"] == "lego-3001"
        assert identify.run_view(seeded, res["run_id"])["activity"][0]["tool"] == "brickognize_identify"
        _brick_mock(m, score=0.2)
        res = identify.identify(seeded, [_jpeg()])
        assert res["match"] == "none" and res["stop_reason"] == "keyless_no_match"


def test_load_rejects_non_images_and_expand_paths(tmp_path):
    with pytest.raises(ValueError):
        identify._load(b"not an image")
    d = tmp_path / "item"
    d.mkdir()
    for n in ("a.jpg", "b.PNG", "notes.txt"):
        (d / n).write_bytes(b"x")
    assert [os.path.basename(p) for p in identify.expand_paths([str(d)])] == ["a.jpg", "b.PNG"]


# ---------------------------------------------------------------- web fetch guard

def test_web_fetch_text_and_guards(seeded, monkeypatch):
    page = "<html><head><title>ESP-01</title><script>evil()</script></head><body><p>Size: 24.8 x 14.3 mm</p></body></html>"
    monkeypatch.setattr(web, "_public_host", lambda h: h != "localhost")
    with track.run(seeded, "test"), respx.mock() as m:
        m.get("https://shop.test/p").mock(return_value=httpx.Response(200, text=page, headers={"content-type": "text/html"}))
        m.get("https://shop.test/r").mock(return_value=httpx.Response(302, headers={"location": "http://localhost/admin"}))
        out = web.fetch_text("https://shop.test/p")
        assert out["title"] == "ESP-01" and "24.8 x 14.3" in out["text"] and "evil" not in out["text"]
        with pytest.raises(ValueError, match="local/private"):
            web.fetch_text("https://shop.test/r")
        with pytest.raises(ValueError):
            web.fetch_text("file:///etc/passwd")


# ---------------------------------------------------------------- dashboard endpoints

def test_identify_endpoints(seeded, no_key):
    from dashboard.app import app
    client = TestClient(app)
    assert client.get("/objects/identify").status_code == 200
    st = client.get("/api/identify/status").json()
    assert st["ready"] is False and "keyless" in st["mode"]
    assert client.post("/api/identify", files=[("files", ("x.txt", b"nope", "text/plain"))]).status_code == 400
    assert client.post("/api/identify", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))],
                       data={"dims": "1 2"}).status_code == 400
    with respx.mock() as m:
        _brick_mock(m)
        r = client.post("/api/identify", files=[("files", ("a.jpg", _jpeg(), "image/jpeg"))], data={"hint": "red brick"})
        assert r.status_code == 202
        rid = r.json()["run_id"]
        for _ in range(100):
            v = client.get(f"/api/identify/{rid}").json()
            if v["status"] != "running":
                break
            time.sleep(0.05)
    assert v["result"]["object_id"] == "lego-3001" and v["object"]["ldraw_id"] == "3001" and v["hint"] == "red brick"
    assert client.post(f"/api/identify/{rid}/confirm", json={"correct": "yes"}).status_code == 400
    c = client.post(f"/api/identify/{rid}/confirm", json={"correct": True, "owned": True}).json()
    assert c["action"] == "confirmed" and c["view"]["sighting"] == "confirmed"
    assert client.post(f"/api/identify/{rid}/confirm", json={"correct": True}).status_code == 409
    assert client.get("/api/identify/nope").status_code == 404
    assert client.get("/api/identify").json()["runs"][0]["run_id"] == rid


# ---------------------------------------------------------------- billing / model errors, search fallback

def test_errors_are_explained_and_search_falls_back(seeded, monkeypatch):
    from objsearch.sources import ddg
    monkeypatch.setattr(ddg, "search", lambda q, n=6: {"results": [{"title": "t", "url": "https://ddg.test/x"}],
                                                        "trust": "fallback"})
    billing = gerrors.ClientError(402, {"error": {"code": 402, "message": "Your prepayment credits are depleted.",
                                                  "status": "RESOURCE_EXHAUSTED"}})
    client = FakeClient({"search-m": [billing], "agent-m": [billing]})
    be = gemini_backend.GeminiBackend("agent-m", "search-m", client=client)
    with track.run(seeded, "test"):
        out1 = be.grounded_search("esp-01")              # grounding fails -> DuckDuckGo
        out2 = be.grounded_search("esp-12")              # stays on the fallback, no second Gemini call
        assert out1["trust"] == "fallback" and "prepaid billing" in out2["note"]
        assert len(client.calls) == 1
        with pytest.raises(gemini_backend.GeminiError, match="prepaid billing"):
            be._generate("agent-m", "x", None)
    gone = gerrors.ClientError(404, {"error": {"code": 404, "message": "This model models/x is no longer available to new users.",
                                               "status": "NOT_FOUND"}})
    assert "models check" in str(gemini_backend.explain(gone))
    assert gemini_backend.explain(gerrors.ClientError(400, {"error": {"code": 400, "message": "bad arg"}})) is None


def test_check_key(monkeypatch):
    ok = FakeClient({"m": [_text("ok")]})
    assert gemini_backend.check_key(ok, "m")["ok"] is True
    billing = FakeClient({"m": [gerrors.ClientError(402, {"error": {"code": 402, "message": "credits depleted"}})]})
    r = gemini_backend.check_key(billing, "m")
    assert r["ok"] is False and r["problem"] == "billing" and "AI Studio" in r["message"]
    monkeypatch.setattr(gemini_backend, "api_key", lambda: None)
    assert gemini_backend.check_key(None, "m")["problem"] == "no_key"


def test_billing_error_reaches_the_result(seeded, monkeypatch):
    billing = gerrors.ClientError(402, {"error": {"code": 402, "message": "Your prepayment credits are depleted."}})
    be = gemini_backend.GeminiBackend("agent-m", "search-m", client=FakeClient({"agent-m": [billing]}))
    res = identify.identify(seeded, [_jpeg()], backend=be)
    assert res["match"] == "none" and "prepaid billing" in res["stop_reason"]


def test_evidence_limited_to_seen_urls(seeded, monkeypatch):
    from objsearch.sources import ddg
    monkeypatch.setattr(ddg, "search", lambda q, n=6: {"results": [{"title": "t", "url": "https://seen.test/uno"}]})
    monkeypatch.setitem(config.load(), "search_mode", "duckduckgo")
    final = {**FINAL, "catalog_id": "", "name": "Odd board", "evidence_urls": ["https://seen.test/uno", "https://memory.test/x"]}
    be = gemini_backend.GeminiBackend("agent-m", "search-m", client=FakeClient({"agent-m": [
        _call("web_search", {"query": "odd board"}), _call("submit_result", final)]}))
    res = identify.identify(seeded, [_jpeg()], backend=be)
    assert res["evidence_urls"] == ["https://seen.test/uno"] and res["unverified_urls"] == ["https://memory.test/x"]
    trust = {e["url"]: e["trust"] for e in db.get_object_full(seeded, res["object_id"])["evidence"]}
    assert trust["https://seen.test/uno"] == "seen by agent" and trust["https://memory.test/x"].startswith("unverified")
