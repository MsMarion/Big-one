"""Scripted LLM for tests and demos: plays a list of turns through the real agent loop machinery
(tool execution, budgets, validation, tracking, transcripts) with no network.

Turn forms:
    {"tool_calls": [{"name": "web_search", "args": {...}}, ...], "text": "optional thinking-out-loud"}
    {"final": {...}}                       # must validate against output_schema
    {"raise": "ConnectionError: 503"}      # provider failure mid-loop
    {"tokens": [in, out]}                  # optional per-turn token counts (merged into any turn)
"""
import builtins
import time

from .. import track
from .base import AgentResult, Backend, BudgetExceeded, execute_tool, save_transcript, validate


class FakeLLM(Backend):
    provider = "fake"

    def __init__(self, script, model="fake"):
        super().__init__(model)
        self.script = list(script)

    def run(self, system, content, tools, output_schema, budget):
        tmap = {t.name: t for t in tools}
        transcript = [{"role": "system", "text": system},
                      {"role": "user", "content": [c if isinstance(c, str) else f"<image {c.label}>" for c in content]}]
        errors = []
        try:
            for turn in self.script:
                budget.charge("llm_calls")
                t0 = time.perf_counter()
                if "raise" in turn:
                    name, _, msg = turn["raise"].partition(": ")
                    exc = getattr(builtins, name, RuntimeError)
                    raise exc(msg or name)
                tin, tout = turn.get("tokens", (400, 80))
                stop = "end_turn" if "final" in turn else "tool_use"
                lid = track.record_llm_call(self.provider, self.model, tin, tout,
                                            round((time.perf_counter() - t0) * 1e3), stop)
                if "final" in turn:
                    transcript.append({"role": "assistant", "final": turn["final"]})
                    problems = validate(output_schema, turn["final"]) if output_schema else []
                    if problems:
                        errors += problems
                        return self._done(None, "invalid_output", transcript, budget, errors)
                    return self._done(turn["final"], "final", transcript, budget, errors)
                calls = turn.get("tool_calls", [])
                transcript.append({"role": "assistant", "text": turn.get("text"), "tool_calls": calls})
                results = []
                for c in calls:
                    res, is_err = execute_tool(tmap, c["name"], c.get("args"), budget, lid)
                    results.append({"name": c["name"], "result": res, "is_error": is_err})
                transcript.append({"role": "tool", "results": results})
        except BudgetExceeded as e:
            errors.append(str(e))
            return self._done(None, "budget_exhausted", transcript, budget, errors)
        except Exception:
            save_transcript(transcript)
            raise
        return self._done(None, "script_exhausted", transcript, budget, errors)

    def _done(self, output, stop, transcript, budget, errors):
        save_transcript(transcript)
        return AgentResult(output, stop, transcript, budget.snapshot(), errors)
