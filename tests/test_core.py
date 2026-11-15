from app.bm25 import BM25Index, tokenize
from app.chunking import chunk_pages, split_sentences
from app.loaders import UnsupportedFile, load
from app.retriever import reciprocal_rank_fusion

import pytest


def test_split_sentences_handles_urdu_full_stop():
    assert split_sentences("One. Two! یہ جملہ ہے۔ Next?") == ["One.", "Two!", "یہ جملہ ہے۔", "Next?"]


def test_chunks_respect_budget_and_overlap():
    para = " ".join(f"Sentence number {i} is here." for i in range(12))
    pages = [(1, "\n\n".join([para] * 3)), (2, para)]
    chunks = chunk_pages(pages, max_chars=400, overlap_chars=80)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 + 80 for c in chunks)
    assert chunks[0].page_start == 1 and chunks[-1].page_end == 2
    # the tail of one chunk opens the next
    assert chunks[0].text.split(". ")[-1].rstrip(".") in chunks[1].text


def test_headings_are_tracked_and_split_chunks():
    text = "# Leave\n\nAnnual leave is 20 days.\n\n# Travel\n\nBook flights through the portal."
    chunks = chunk_pages([(1, text)])
    assert [(c.heading, c.text) for c in chunks] == [
        ("Leave", "Annual leave is 20 days."),
        ("Travel", "Book flights through the portal."),
    ]


def test_oversized_sentence_is_hard_split():
    chunks = chunk_pages([(1, "x" * 2500)], max_chars=1000, overlap_chars=0)
    assert [len(c.text) for c in chunks] == [1000, 1000, 500]


def test_loader_rejects_unknown_and_empty():
    with pytest.raises(UnsupportedFile):
        load("virus.exe", b"MZ")
    with pytest.raises(UnsupportedFile):
        load("empty.txt", b"   \n ")
    assert load("a.md", b"hello-\nworld\r\n\r\n\r\n\r\nbye") == [(1, "helloworld\n\nbye")]


def test_tokenize_keeps_codes_drops_stopwords_folds_plurals():
    assert tokenize("Error E-4012 on v2.1") == ["error", "e-4012", "e", "4012", "v2.1", "v2", "1"]
    assert tokenize("Can I ship the parcels?") == ["ship", "parcel"]
    assert tokenize("Suspicious emails") == tokenize("suspicious email")


def test_bm25_ranks_exact_code_first_and_filters_and_removes():
    index = BM25Index()
    index.add("a", "d1", "Error E-4012 means the parcel weight is missing.")
    index.add("b", "d1", "General information about parcels and weight limits for parcels.")
    index.add("c", "d2", "Error E-4012 is also listed in the legacy guide.")
    assert index.search("what is E-4012", 5)[0][0] in {"a", "c"}
    assert {cid for cid, _ in index.search("E-4012", 5)} == {"a", "c"}
    assert [cid for cid, _ in index.search("E-4012", 5, doc_ids={"d2"})] == ["c"]
    index.remove_document("d1")
    assert len(index) == 1
    assert [cid for cid, _ in index.search("E-4012 parcels", 5)] == ["c"]
    assert index.search("nothing matches this zzz", 5) == []


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "d", "a"]], k=60)
    assert [item for item, _ in fused][:2] == ["b", "a"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)
