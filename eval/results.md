Corpus: 7 documents, 52 chunks (350-char chunks). Questions: 36.
Embeddings: BAAI/bge-small-en-v1.5. Reranker: Xenova/ms-marco-MiniLM-L-6-v2.

| Mode | Hit@1 | Hit@3 | MRR | ms / query |
|---|---|---|---|---|
| dense | 78% | 89% | 0.834 | 4 |
| bm25 | 50% | 64% | 0.586 | 0 |
| hybrid | 61% | 86% | 0.732 | 4 |
| hybrid_rerank | 86% | 94% | 0.910 | 135 |
