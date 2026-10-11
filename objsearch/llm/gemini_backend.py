"""Gemini (google-genai SDK), free tier by default.

Manual function-calling loop: our tools are FunctionDeclarations, the model's turns are appended back
unchanged (Gemini 3 thought signatures must round-trip), and the final answer arrives as a call to
`submit_result` whose parameters are the output schema (works with tools, no response_schema needed).

`web_search` is a sub-call to the search model with Google Search grounding. Grounding on Gemini 3.x
needs a billed (paid-tier) project (5,000 searches/month included); the 2.5 models that had free
grounding are closed to new users. When grounding fails (free tier, quota, billing) web_search falls
back to keyless DuckDuckGo for the rest of the run. Every call checks the local quota and is tracked.
API errors are turned into GeminiError with a plain-English fix (402 billing, 404 model, bad key).
"""
import json
import os
import time

from .. import config, track
from .base import AgentResult, Backend, BudgetExceeded, execute_tool, save_transcript, validate

SUBMIT = "submit_result"


class GeminiError(RuntimeError):
    """A Gemini API problem the user has to fix (billing, model access, key)."""


def explain(e):
    """APIError -> GeminiError with what to do about it (or None for transient errors)."""
    msg = (getattr(e, "message", None) or str(e))[:300]
    if e.code == 402:
        return GeminiError(
            "Gemini refused the request: this key's Google project has prepaid billing with no credits left (402). "
            "Either add credits in AI Studio (Projects > Billing), or create a NEW project there WITHOUT billing and "
            "use a key from it for the free tier.")
    if e.code == 404 and "no longer available" in msg:
        return GeminiError(f"Model not available to this key: {msg} Run `python -m objsearch models check` and "
                           f"update models.gemini in configs/objsearch.yaml.")
    if e.code in (401, 403) or (e.code == 400 and "API key" in msg):
        return GeminiError(f"The Gemini key was rejected ({e.code}): {msg} Check GEMINI_API_KEY in .env.")
    return None


def api_key():
    config.load_dotenv()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def make_client():
    from google import genai
    key = api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (free key: https://aistudio.google.com/apikey, put it in .env)")
    return genai.Client(api_key=key)


class GeminiBackend(Backend):
    provider = "gemini"

    def __init__(self, model=None, search_model=None, client=None, thinking_level=None):
        m = config.load()["models"]["gemini"]
        super().__init__(model or m["agent"])
        self.search_model = search_model or m["search"]
        self._client = client
        self.thinking_level = thinking_level  # e.g. "LOW" for fast tool-driven jobs (Gemini 3)

    @property
    def client(self):
        if self._client is None:
            self._client = make_client()
        return self._client

    # ------------------------------------------------------------ calls

    def _generate(self, model, contents, cfg, quota_kind="request"):
        from google.genai import errors
        conn = track.current_run().conn
        q = config.load().get("quota", {})
        limit = q.get("grounded_search_per_day") if quota_kind == "grounded_search" else \
            q.get("requests_per_day", {}).get(model)
        track.check_quota(conn, quota_kind, limit, None if quota_kind == "grounded_search" else model)
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=cfg)
            except errors.APIError as e:
                ms = round((time.perf_counter() - t0) * 1e3)
                track.record_llm_call(self.provider, model, 0, 0, ms, f"error {e.code}", quota_kind=quota_kind)
                if e.code == 429 and attempt < 2:
                    track.log.warning(f"{model} rate limited (429); retrying in {10 * (attempt + 1)} s")
                    time.sleep(10 * (attempt + 1))
                    continue
                if e.code == 429:
                    raise track.QuotaExceeded(f"{model}: free-tier rate limit hit (429)") from e
                raise (explain(e) or e) from e
            u = resp.usage_metadata
            tin = (u.prompt_token_count or 0) if u else 0
            tout = ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0
            cand = resp.candidates[0] if resp.candidates else None
            stop = str(cand.finish_reason.value if cand and cand.finish_reason else "none")
            lid = track.record_llm_call(self.provider, model, tin, tout, round((time.perf_counter() - t0) * 1e3),
                                        stop, request_id=getattr(resp, "response_id", None), quota_kind=quota_kind)
            return resp, lid

    def grounded_search(self, query):
        """web_search tool: Google-grounded answer + sources; DuckDuckGo if grounding isn't available."""
        mode = config.load().get("search_mode", "auto")
        if mode == "duckduckgo" or (mode == "auto" and getattr(self, "_grounding_off", None)):
            from ..sources import ddg
            out = ddg.search(query)
            if getattr(self, "_grounding_off", None):
                out["note"] = f"Google grounding unavailable this run: {self._grounding_off}"
            return out
        try:
            return self._grounded(query)
        except Exception as e:  # free tier / billing / quota: switch to the fallback for the rest of the run
            if mode == "grounded":
                raise
            self._grounding_off = f"{type(e).__name__}: {str(e)[:160]}"
            track.log.warning(f"grounded search unavailable, using DuckDuckGo: {self._grounding_off}")
            return self.grounded_search(query)

    def _grounded(self, query):
        from google.genai import types
        cfg = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        prompt = (f"Search the web for: {query}\nAnswer in a few factual sentences (product names, part "
                  f"numbers, dimensions in mm) and rely only on what the search results say.")
        resp, _ = self._generate(self.search_model, prompt, cfg, quota_kind="grounded_search")
        sources = []
        cand = resp.candidates[0] if resp.candidates else None
        gm = cand.grounding_metadata if cand else None
        for ch in (gm.grounding_chunks or []) if gm else []:
            if ch.web and ch.web.uri:
                sources.append({"title": ch.web.title or ch.web.domain, "url": ch.web.uri})
        return {"answer": (resp.text or "")[:3000], "sources": sources[:8],
                "queries": list(gm.web_search_queries or []) if gm else [], "trust": "grounded"}

    # ------------------------------------------------------------ agent loop

    def run(self, system, content, tools, output_schema, budget):
        from google.genai import types
        tmap = {t.name: t for t in tools}
        decls = [types.FunctionDeclaration(name=t.name, description=t.description, parameters_json_schema=t.parameters)
                 for t in tools]
        decls.append(types.FunctionDeclaration(
            name=SUBMIT, description="Submit the final answer. Call exactly once, when you are done.",
            parameters_json_schema=output_schema))
        cfg = types.GenerateContentConfig(
            system_instruction=system, tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None)
        parts, shown = [], []
        for c in content:
            if isinstance(c, str):
                parts.append(types.Part.from_text(text=c))
                shown.append(c)
            else:
                parts.append(types.Part.from_text(text=f"[image {c.label}]"))
                parts.append(types.Part.from_bytes(data=c.bytes(), mime_type=c.mime))
                shown.append(f"<image {c.label}>")
        contents = [types.Content(role="user", parts=parts)]
        transcript = [{"role": "system", "text": system}, {"role": "user", "content": shown}]
        errors, nudged = [], False
        try:
            while True:
                budget.charge("llm_calls")
                resp, lid = self._generate(self.model, contents, cfg)
                cand = resp.candidates[0] if resp.candidates else None
                if cand is None or cand.content is None or not cand.content.parts:
                    reason = str(cand.finish_reason) if cand else "no candidates"
                    errors.append(f"empty response ({reason})")
                    return self._done(None, "empty_response", transcript, budget, errors)
                contents.append(cand.content)  # unchanged: keeps thought signatures
                text = "".join(p.text for p in cand.content.parts if p.text and not p.thought)
                calls = resp.function_calls or []
                transcript.append({"role": "assistant", "text": text or None,
                                   "tool_calls": [{"name": fc.name, "args": dict(fc.args or {})} for fc in calls]})
                if not calls:
                    if nudged:
                        errors.append("model answered in text instead of calling submit_result")
                        return self._done(None, "invalid_output", transcript, budget, errors)
                    nudged = True
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(
                        text=f"Call {SUBMIT} now with your final answer.")]))
                    continue
                replies, results = [], []
                for fc in calls:
                    args = json.loads(json.dumps(dict(fc.args or {}), default=str))
                    if fc.name == SUBMIT:
                        problems = validate(output_schema, args)
                        if not problems:
                            transcript.append({"role": "assistant", "final": args})
                            return self._done(args, "final", transcript, budget, errors)
                        errors += problems
                        res, is_err = {"error": "invalid result: " + "; ".join(problems)}, True
                        track.record_tool_call(SUBMIT, args, res, True, 0, lid)
                    else:
                        res, is_err = execute_tool(tmap, fc.name, args, budget, lid)
                    results.append({"name": fc.name, "result": res, "is_error": is_err})
                    replies.append(types.Part.from_function_response(
                        name=fc.name, response=res if isinstance(res, dict) else {"result": res}))
                transcript.append({"role": "tool", "results": results})
                contents.append(types.Content(role="user", parts=replies))
        except BudgetExceeded as e:
            errors.append(str(e))
            return self._done(None, "budget_exhausted", transcript, budget, errors)
        except Exception:
            save_transcript(transcript)
            raise

    def _done(self, output, stop, transcript, budget, errors):
        save_transcript(transcript)
        return AgentResult(output, stop, transcript, budget.snapshot(), errors)


def list_models(client=None):
    c = client or make_client()
    return sorted(m.name.removeprefix("models/") for m in c.models.list())


def check_key(client=None, model=None):
    """One tiny request to the agent model: is the key usable right now? -> dict for CLI/dashboard."""
    from google.genai import errors
    model = model or config.load()["models"]["gemini"]["agent"]
    try:
        c = client or make_client()
    except RuntimeError as e:
        return {"ok": False, "model": model, "problem": "no_key", "message": str(e)}
    try:
        r = c.models.generate_content(model=model, contents="Reply with the single word: ok")
        return {"ok": True, "model": model, "message": f"{model} answered: {(r.text or '').strip()[:20]!r}"}
    except errors.APIError as e:
        friendly = explain(e)
        kind = {402: "billing", 404: "model", 429: "rate_limit"}.get(e.code, "key" if e.code in (400, 401, 403) else "api")
        return {"ok": False, "model": model, "problem": kind, "code": e.code,
                "message": str(friendly) if friendly else f"{e.code} {e.status}: {(e.message or '')[:200]}"}
