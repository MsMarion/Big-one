"""Provider-agnostic agent interface. Backends (gemini, anthropic, fake) implement Backend.run();
tool execution, budgets, output validation and transcripts are shared here so every backend is
tracked the same way."""
import abc
import dataclasses
import json
import time
from typing import Any, Callable

from .. import track

SEARCH_TOOLS = {"web_search"}


@dataclasses.dataclass
class Image:
    path: str | None = None
    data: bytes | None = None
    mime: str = "image/jpeg"
    label: str = ""              # how the prompt refers to it, e.g. "crop_0"

    def bytes(self):
        if self.data is None:
            with open(self.path, "rb") as f:
                self.data = f.read()
        return self.data


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    parameters: dict             # JSON schema (object); kept to the subset Gemini and Claude both accept
    handler: Callable[[dict], Any]


class BudgetExceeded(Exception):
    pass


class StopAgent(BudgetExceeded):
    """Raised by a tool once it has found what the run is looking for: ends the loop without another LLM call
    (the caller reads the result from its own state)."""


@dataclasses.dataclass
class Budget:
    max_llm_calls: int = 20
    max_tool_calls: int = 40
    max_searches: int = 8
    llm_calls: int = 0
    tool_calls: int = 0
    searches: int = 0

    @classmethod
    def from_config(cls, cfg):
        b = cfg.get("budget", {})
        return cls(**{k: b[k] for k in ("max_llm_calls", "max_tool_calls", "max_searches") if k in b})

    def charge(self, kind):
        n = getattr(self, kind) + 1
        if n > getattr(self, f"max_{kind}"):
            raise BudgetExceeded(f"{kind} budget exhausted ({getattr(self, f'max_{kind}')})")
        setattr(self, kind, n)

    def snapshot(self):
        return {"llm_calls": self.llm_calls, "tool_calls": self.tool_calls, "searches": self.searches}


@dataclasses.dataclass
class AgentResult:
    output: dict | None
    stop_reason: str             # final|budget_exhausted|invalid_output|script_exhausted|max_tokens|refusal
    transcript: list
    budget: dict
    errors: list = dataclasses.field(default_factory=list)


class Backend(abc.ABC):
    provider = "base"

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def run(self, system: str, content: list, tools: list[Tool], output_schema: dict,
            budget: Budget) -> AgentResult:
        """Loop: model -> tool calls (via execute_tool) -> ... -> final JSON matching output_schema."""


def execute_tool(tools: dict, name, args, budget: Budget, llm_call_id=None):
    """Run one tool call with tracking. Never raises for tool failures: the model gets an error result."""
    t0 = time.perf_counter()
    try:
        budget.charge("tool_calls")
        if name in SEARCH_TOOLS:
            budget.charge("searches")
        if name not in tools:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(tools)}")
        if not isinstance(args, dict):
            raise TypeError(f"tool input must be a JSON object, got {type(args).__name__}")
        problems = validate(tools[name].parameters, args)
        if problems:
            raise ValueError("invalid tool input: " + "; ".join(problems))
        result, is_error = tools[name].handler(args), False
    except StopAgent as e:
        track.record_tool_call(name, args, f"stop: {e}", False, round((time.perf_counter() - t0) * 1e3), llm_call_id)
        raise
    except BudgetExceeded:
        raise
    except Exception as e:  # tool bugs and bad inputs go back to the model as an error result
        result, is_error = {"error": f"{type(e).__name__}: {e}"}, True
    ms = round((time.perf_counter() - t0) * 1e3)
    track.record_tool_call(name, args, summarize(result), is_error, ms, llm_call_id)
    return result, is_error


def summarize(result, n=600):
    s = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + f"… (+{len(s) - n} chars)"


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(schema, obj, path="$"):
    """Small JSON-schema check (type/required/enum/items/properties) shared by tool inputs and outputs."""
    errs = []
    t = schema.get("type")
    if t and t in _TYPES:
        ok = isinstance(obj, _TYPES[t]) and not (t in ("integer", "number") and isinstance(obj, bool))
        if not ok:
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        errs.append(f"{path}: {obj!r} not in {schema['enum']}")
    if t == "object":
        for k in schema.get("required", []):
            if k not in obj:
                errs.append(f"{path}.{k}: required")
        for k, sub in schema.get("properties", {}).items():
            if k in obj and obj[k] is not None:
                errs += validate(sub, obj[k], f"{path}.{k}")
    if t == "array" and "items" in schema:
        for i, x in enumerate(obj):
            errs += validate(schema["items"], x, f"{path}[{i}]")
    return errs


def save_transcript(transcript):
    """Write the transcript of the current agent step to the run dir; returns the data-relative path."""
    s, r = track.current_step(), track.current_run()
    if r is None:
        return None
    p = r.path("transcripts", f"step_{s.id if s else 'x'}.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for t in transcript:
            f.write(json.dumps(track.redact(t), ensure_ascii=False, default=str) + "\n")
    rel = track.rel(p)
    if s:
        s.set(transcript=rel)
    return rel
