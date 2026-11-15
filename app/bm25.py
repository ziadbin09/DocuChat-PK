"""In-memory Okapi BM25 index for the keyword half of hybrid search.

Dense vectors miss exact identifiers (error codes, form numbers, names); BM25
catches them. The index is rebuilt from the vector store's payloads at startup.
"""
import math
import re
from collections import Counter, defaultdict

TOKEN = re.compile(r"\w+(?:[-./]\w+)*", re.UNICODE)

# Function words carry no topic signal; left in, they let a question like
# "Can I ship a necklace?" match every chunk that contains "I" or "a".
STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have how i if in "
    "into is it its me my no not of on or our she he so than that the their them then there these "
    "they this to up us was we were what when where which who why will with would you your".split()
)


def _normalise(token: str) -> str:
    # Fold simple plurals so "emails" matches "email". Deliberately tiny: no stemmer dependency.
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in TOKEN.findall(text.lower()):
        parts = re.split(r"[-./]", tok)
        if len(parts) > 1:  # keep "e-4012" whole, and also index "4012"
            tokens.append(tok)
            tokens.extend(_normalise(p) for p in parts if p and p not in STOPWORDS)
        elif tok not in STOPWORDS:
            tokens.append(_normalise(tok))
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        self._doc_of: dict[str, str] = {}
        self._total_len = 0

    def __len__(self) -> int:
        return len(self._lengths)

    def add(self, chunk_id: str, doc_id: str, text: str) -> None:
        if chunk_id in self._lengths:
            return
        tokens = tokenize(text)
        for term, tf in Counter(tokens).items():
            self._postings[term][chunk_id] = tf
        self._lengths[chunk_id] = len(tokens)
        self._doc_of[chunk_id] = doc_id
        self._total_len += len(tokens)

    def remove_document(self, doc_id: str) -> None:
        gone = {cid for cid, d in self._doc_of.items() if d == doc_id}
        if not gone:
            return
        for term in list(self._postings):
            posting = self._postings[term]
            for cid in gone & posting.keys():
                del posting[cid]
            if not posting:
                del self._postings[term]
        for cid in gone:
            self._total_len -= self._lengths.pop(cid)
            del self._doc_of[cid]

    def search(self, query: str, limit: int, doc_ids: set[str] | None = None) -> list[tuple[str, float]]:
        n = len(self._lengths)
        if not n:
            return []
        avg_len = self._total_len / n or 1.0
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for cid, tf in posting.items():
                if doc_ids is not None and self._doc_of[cid] not in doc_ids:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self._lengths[cid] / avg_len)
                scores[cid] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
