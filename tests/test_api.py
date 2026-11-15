"""End-to-end API tests with a fake LLM and hashing embeddings: no network, no model downloads."""
import json

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashEmbedder
from app.llm import LLMError
from app.main import create_app

HANDBOOK = b"""# Leave

Employees receive 20 days of annual leave per year. Unused leave carries over up to 5 days.

# Expenses

Submit expense claims on form FIN-27 within 30 days. Claims above 50,000 PKR need director approval.
"""


class FakeLLM:
    model = "fake-model"

    def __init__(self):
        self.calls = []
        self.fail = False

    async def rewrite(self, question, history):
        return f"standalone: {question} (about {history[0]['content']})"

    async def stream_answer(self, question, sources, history):
        self.calls.append({"question": question, "sources": sources, "history": history})
        if self.fail:
            raise LLMError("The model is rate limited right now. Try again in a minute.")
        yield {"type": "text", "text": "Answer from "}
        yield {"type": "text", "text": sources[0]["doc_name"]}
        yield {"type": "citation", "n": 1, "quote": sources[0]["text"][:20]}
        yield {"type": "text", "text": " [1]"}
        yield {"type": "end", "stop_reason": "end_turn", "model": self.model,
               "usage": {"input_tokens": 10, "output_tokens": 5}}


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def client(llm):
    app = create_app(Settings(_env_file=None), embedder=HashEmbedder(), llm=llm, persist=False)
    with TestClient(app) as c:
        yield c


def sse(response):
    events = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def upload(client, name="handbook.md", data=HANDBOOK):
    return client.post("/api/documents", files=[("files", (name, data, "text/markdown"))]).json()[0]


def test_upload_list_dedupe_delete(client):
    first = upload(client)
    assert first["status"] == "indexed" and first["document"]["chunks"] == 2
    assert upload(client)["status"] == "duplicate"
    assert upload(client, "bad.exe", b"x")["status"] == "error"
    assert [d["name"] for d in client.get("/api/documents").json()] == ["handbook.md"]
    assert client.get("/api/health").json()["chunks"] == 2

    doc_id = first["document"]["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 200
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404
    assert client.get("/api/health").json()["chunks"] == 0
    assert client.post("/api/search", json={"query": "annual leave"}).json()["results"] == []


def test_search_finds_exact_code_and_respects_doc_filter(client):
    doc = upload(client)["document"]
    other = upload(client, "menu.txt", b"The cafeteria serves biryani on Fridays.")["document"]

    results = client.post("/api/search", json={"query": "FIN-27 claim form"}).json()["results"]
    assert results[0]["doc_id"] == doc["id"]
    assert "FIN-27" in results[0]["text"] and results[0]["location"] == "p.1, Expenses"
    assert results[0]["scores"]["bm25_rank"] == 1

    scoped = client.post("/api/search", json={"query": "FIN-27 claim form", "doc_ids": [other["id"]]}).json()
    assert {r["doc_id"] for r in scoped["results"]} <= {other["id"]}
    assert client.post("/api/search", json={"query": "x", "mode": "nope"}).status_code == 422


def test_chat_streams_sources_tokens_citations(client, llm):
    upload(client)
    events = sse(client.post("/api/chat", json={"question": "How many days of annual leave?"}))
    names = [n for n, _ in events]
    assert names[0] == "status" and names[-1] == "done"
    assert names.index("sources") < names.index("token")

    data = dict(events)
    assert data["sources"]["query"] == "How many days of annual leave?"
    assert "annual leave" in data["sources"]["sources"][0]["text"]
    assert "".join(d["text"] for n, d in events if n == "token") == "Answer from handbook.md [1]"
    assert data["citation"]["n"] == 1
    assert data["done"]["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert llm.calls[0]["history"] == []


def test_chat_rewrites_follow_ups_and_cleans_history(client, llm):
    upload(client)
    history = [
        {"role": "assistant", "content": "stray leading assistant turn"},
        {"role": "user", "content": "expenses"},
        {"role": "assistant", "content": "Use form FIN-27."},
    ]
    events = dict(sse(client.post("/api/chat", json={"question": "what is the deadline?", "history": history})))
    assert events["sources"]["query"] == "standalone: what is the deadline? (about expenses)"
    assert [m["role"] for m in llm.calls[0]["history"]] == ["user", "assistant"]
    assert llm.calls[0]["question"] == "what is the deadline?"  # the model sees the original wording


def test_chat_without_documents_and_llm_errors(client, llm):
    events = sse(client.post("/api/chat", json={"question": "anything"}))
    assert dict(events)["done"]["stop_reason"] == "no_sources" and not llm.calls

    upload(client)
    llm.fail = True
    events = sse(client.post("/api/chat", json={"question": "annual leave"}))
    assert events[-1] == ("error", {"message": "The model is rate limited right now. Try again in a minute."})
    assert client.post("/api/chat", json={"question": ""}).status_code == 422
