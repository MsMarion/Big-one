"""LLM backends behind one interface (llm.base.Backend): gemini (free tier, default), anthropic, fake."""


def get_backend(provider, model=None, **kw):
    if provider == "fake":
        from .fake_backend import FakeLLM
        return FakeLLM(kw.pop("script", []), model=model or "fake")
    if provider == "gemini":
        from .gemini_backend import GeminiBackend
        return GeminiBackend(model, client=kw.pop("client", None))
    raise NotImplementedError(f"provider {provider!r} arrives in a later phase (phase 5: anthropic)")
