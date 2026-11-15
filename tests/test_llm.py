"""AnthropicLLM against a fake SDK client: checks the request we build and how
streamed text and citation events are translated. No network."""
import asyncio
from types import SimpleNamespace as NS

import pytest

from app.config import Settings
from app.llm import AnthropicLLM, LLMError

SOURCES = [
    {"n": 1, "title": "api.md (p.1)", "uri": "doc://a#chunk-0", "doc_name": "api.md",
     "text": "E-4012 means the weight is missing. E-5000 is a server error."},
    {"n": 2, "title": "faq.md (p.1)", "uri": "doc://b#chunk-3", "doc_name": "faq.md",
     "text": "The maximum weight is 30 kg."},
]


def delta(**kw):
    return NS(type="content_block_delta", delta=NS(**kw))


def citation(index, text):
    return delta(type="citations_delta",
                 citation=NS(type="search_result_location", search_result_index=index, cited_text=text))


class FakeStream:
    def __init__(self, events, error=None):
        self.events, self.error = events, error

    async def __aenter__(self):
        if self.error:
            raise self.error
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()

    async def get_final_message(self):
        return NS(stop_reason="end_turn", model="claude-opus-5-5",
                  usage=NS(input_tokens=120, output_tokens=30))


class FakeMessages:
    def __init__(self, events=(), error=None):
        self.events, self.error, self.kwargs = list(events), error, None

    def stream(self, **kwargs):
        self.kwargs = kwargs
        return FakeStream(self.events, self.error)


def make_llm(events=(), error=None, **settings):
    llm = AnthropicLLM(Settings(_env_file=None, anthropic_api_key="test-key", **settings))
    beta, plain = FakeMessages(events, error), FakeMessages(events, error)
    llm.client = NS(beta=NS(messages=beta), messages=plain)
    return llm, beta, plain


def collect(llm, history=()):
    async def run():
        return [e async for e in llm.stream_answer("What is E-4012?", SOURCES, list(history))]
    return asyncio.run(run())


def test_request_sends_chunks_as_search_results_with_sentence_blocks():
    llm, beta, plain = make_llm()
    collect(llm, history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    kw = beta.kwargs
    assert plain.kwargs is None
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "medium"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "thinking" not in kw and "temperature" not in kw
    assert [m["role"] for m in kw["messages"]] == ["user", "assistant", "user"]
    first, second, question = kw["messages"][-1]["content"]
    assert first == {
        "type": "search_result", "source": "doc://a#chunk-0", "title": "api.md (p.1)",
        "content": [{"type": "text", "text": "E-4012 means the weight is missing."},
                    {"type": "text", "text": "E-5000 is a server error."}],
        "citations": {"enabled": True},
    }
    assert second["title"] == "faq.md (p.1)"
    assert question == {"type": "text", "text": "What is E-4012?"}


def test_fallback_can_be_switched_off():
    llm, beta, plain = make_llm(anthropic_refusal_fallback=False)
    collect(llm)
    assert beta.kwargs is None and "fallbacks" not in plain.kwargs and "betas" not in plain.kwargs


def test_citations_become_markers_at_the_end_of_the_cited_block():
    events = [
        NS(type="content_block_start"),
        delta(type="text_delta", text="It means the weight "),
        delta(type="text_delta", text="is missing"),
        citation(0, "E-4012 means the weight is missing."),
        citation(0, "E-5000 is a server error."),   # same source twice -> one marker
        citation(1, "The maximum weight is 30 kg."),
        NS(type="content_block_stop"),
        delta(type="text_delta", text=". Anything else?"),
        NS(type="content_block_stop"),              # block without citations -> no marker
    ]
    llm, _, _ = make_llm(events)
    out = collect(llm)
    assert "".join(e["text"] for e in out if e["type"] == "text") == \
        "It means the weight is missing [1][2]. Anything else?"
    assert [(e["n"], e["quote"]) for e in out if e["type"] == "citation"] == [
        (1, "E-4012 means the weight is missing."), (2, "The maximum weight is 30 kg."),
    ]
    assert out[-1] == {"type": "end", "stop_reason": "end_turn", "model": "claude-opus-5-5",
                       "usage": {"input_tokens": 120, "output_tokens": 30}}


def test_missing_credentials_become_a_readable_error():
    llm, _, _ = make_llm(error=TypeError("Could not resolve authentication method."))
    with pytest.raises(LLMError, match="No Anthropic API key"):
        collect(llm)
    llm, _, _ = make_llm(error=TypeError("something else"))
    with pytest.raises(TypeError):
        collect(llm)
