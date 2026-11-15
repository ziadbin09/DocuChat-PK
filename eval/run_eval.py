"""Retrieval evaluation: how often does each search mode put the right passage on top?

    python -m eval.run_eval            # real embeddings + reranker (downloads models once)
    python -m eval.run_eval --chunk 350  # smaller chunks = harder retrieval problem

It indexes eval/corpus in memory, runs every question in eval/golden.jsonl through
each retrieval mode, and reports hit@1, hit@3 and MRR. No LLM or API key is needed.
"""
import argparse
import json
import time
from pathlib import Path

from app.config import Settings
from app.embeddings import build_embedder
from app.retriever import MODES, CrossEncoderReranker
from app.service import RagService

HERE = Path(__file__).parent


class _NoLLM:
    model = "none"


def evaluate(service: RagService, golden: list[dict], mode: str, k: int = 5) -> dict:
    hit1 = hit3 = 0
    reciprocal = 0.0
    misses = []
    started = time.perf_counter()
    for case in golden:
        hits = service.search(case["q"], top_k=k, mode=mode)
        rank = next(
            (i for i, h in enumerate(hits, start=1)
             if h.payload["doc_name"] == case["doc"] and case["contains"] in h.text),
            None,
        )
        if rank == 1:
            hit1 += 1
        if rank and rank <= 3:
            hit3 += 1
        if rank:
            reciprocal += 1 / rank
        else:
            misses.append(case["q"])
    n = len(golden)
    return {
        "mode": mode, "hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": reciprocal / n,
        "ms_per_query": (time.perf_counter() - started) * 1000 / n, "misses": misses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=350, help="chunk size in characters")
    parser.add_argument("--hash", action="store_true", help="use hashing embeddings (no downloads)")
    args = parser.parse_args()

    settings = Settings(_env_file=None, chunk_chars=args.chunk, chunk_overlap_chars=60)
    embedder = build_embedder("hash" if args.hash else settings.embed_provider, settings.embed_model)
    reranker = None if args.hash else CrossEncoderReranker(settings.rerank_model)
    service = RagService(settings, embedder, _NoLLM(), reranker, persist=False)

    for path in sorted((HERE / "corpus").glob("*.md")):
        service.ingest(path.name, path.read_bytes())
    golden = [json.loads(line) for line in (HERE / "golden.jsonl").read_text("utf-8").splitlines() if line.strip()]

    # A golden answer that no chunk contains would be an unfair question.
    chunks = [(p["doc_name"], p["text"]) for _, p in service.store.scroll_all()]
    for case in golden:
        if not any(d == case["doc"] and case["contains"] in t for d, t in chunks):
            raise SystemExit(f"Golden passage not found in any chunk: {case['q']!r}")

    modes = [m for m in MODES if reranker or m != "hybrid_rerank"]
    results = [evaluate(service, golden, mode) for mode in modes]

    lines = [
        f"Corpus: {len(service.registry.list())} documents, {len(service.bm25)} chunks "
        f"({args.chunk}-char chunks). Questions: {len(golden)}.",
        f"Embeddings: {'hash (dev)' if args.hash else settings.embed_model}. "
        f"Reranker: {settings.rerank_model if reranker else 'none'}.",
        "",
        "| Mode | Hit@1 | Hit@3 | MRR | ms / query |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r['mode']} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['mrr']:.3f} | {r['ms_per_query']:.0f} |")
    report = "\n".join(lines)
    print(report)
    for r in results:
        if r["misses"]:
            print(f"\n{r['mode']} missed (not in top 5):")
            for q in r["misses"]:
                print(f"  - {q}")
    (HERE / "results.md").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
