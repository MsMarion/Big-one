import json

import pytest

from objsearch import db, track
from objsearch.llm import tools
from objsearch.llm.base import Budget, Tool, validate
from objsearch.llm.fake_backend import FakeLLM

ALLOWED_KEYS = {"type", "properties", "required", "description", "enum", "items"}
ECHO = Tool("echo", "echo", {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            lambda a: {"echo": a["x"]})
OUT = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _walk(schema, path="$"):
    assert set(schema) <= ALLOWED_KEYS, (path, set(schema) - ALLOWED_KEYS)
    for k, sub in schema.get("properties", {}).items():
        _walk(sub, f"{path}.{k}")
    if "items" in schema:
        _walk(schema["items"], f"{path}[]")
    assert set(schema.get("required", [])) <= set(schema.get("properties", {}))


def test_tool_schemas_are_portable():
    """Gemini function declarations and Claude tools both accept this JSON-schema subset."""
    for name, (desc, params) in tools.SCHEMAS.items():
        assert desc and params["type"] == "object"
        _walk(params, name)
        json.dumps(params)
    _walk(tools.IDENTIFICATION_SCHEMA)
    _walk(tools.ACQUIRE_SCHEMA)
    assert set(tools.IDENTIFY_TOOLS) <= set(tools.SCHEMAS) and set(tools.ACQUIRE_TOOLS) <= set(tools.SCHEMAS)
    built = tools.build_tools({"web_search": lambda a: a, "unknown_tool": lambda a: a})
    assert [t.name for t in built] == ["web_search"]


def test_validate():
    s = tools.IDENTIFICATION_SCHEMA
    assert validate(s, {"name": "x", "category": "lego", "confidence": 0.5, "evidence_urls": [], "catalog_id": ""}) == []
    errs = validate(s, {"name": 3, "category": "spaceship", "evidence_urls": "nope"})
    assert any("confidence: required" in e for e in errs) and any("not in" in e for e in errs)
    assert validate({"type": "integer"}, True)  # bools are not integers


def _run(conn, script, budget=None, tools_=(ECHO,)):
    with track.run(conn, "test"):
        with track.step("agent"):
            return FakeLLM(script).run("sys", ["hi"], list(tools_), OUT, budget or Budget())


def test_fake_llm_paths(conn):
    ok = _run(conn, [{"tool_calls": [{"name": "echo", "args": {"x": "a"}}]}, {"final": {"ok": True}}])
    assert ok.stop_reason == "final" and ok.output == {"ok": True} and ok.budget["tool_calls"] == 1
    bad = _run(conn, [{"final": {"ok": "yes"}}])
    assert bad.stop_reason == "invalid_output" and bad.output is None
    out = _run(conn, [{"tool_calls": [{"name": "echo", "args": {"x": "a"}}]}] * 5, Budget(max_tool_calls=2))
    assert out.stop_reason == "budget_exhausted"
    assert _run(conn, []).stop_reason == "script_exhausted"
    with pytest.raises(ConnectionError):
        _run(conn, [{"raise": "ConnectionError: 503"}])


def test_tool_errors_go_back_to_the_model(conn):
    res = _run(conn, [{"tool_calls": [{"name": "nope", "args": {}}, {"name": "echo", "args": "notadict"},
                                      {"name": "echo", "args": {}}, {"name": "echo", "args": {"x": "fine"}}]},
                      {"final": {"ok": True}}])
    results = res.transcript[-2]["results"]
    assert [r["is_error"] for r in results] == [True, True, True, False]
    assert "unknown tool" in results[0]["result"]["error"] and "required" in results[2]["result"]["error"]
    calls = db.rows(conn.execute("SELECT tool, is_error FROM tool_calls ORDER BY id"))
    assert [c["is_error"] for c in calls[-4:]] == [1, 1, 1, 0]


def test_search_budget(conn):
    st = Tool("web_search", "s", tools.SCHEMAS["web_search"][1], lambda a: {"results": []})
    res = _run(conn, [{"tool_calls": [{"name": "web_search", "args": {"query": "q"}}]}] * 4,
               Budget(max_searches=2), tools_=(st,))
    assert res.stop_reason == "budget_exhausted" and res.budget["searches"] == 2


def test_catalog_lookup_handler(seeded):
    h = tools.catalog_lookup_handler(seeded)
    assert h({"text": "2x4 brick"})["matches"][0]["id"] == "lego-3001"
