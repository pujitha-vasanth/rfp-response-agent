"""
The capability library: what our company can and cannot do.

Each entry has an id, a title, a statement (what we offer) and limits
(what we do not offer). The Matcher agent searches this library to find
evidence for each RFP requirement.

Search uses BM25, a classic keyword ranking method. The library is small
(a few dozen entries), so keyword search is fast, free and easy to explain.
For a library with thousands of entries you would switch to vector search
(for example Azure AI Search) without changing the agents.
"""

import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rfp_agent import config

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has", "have",
    "in", "is", "it", "its", "must", "of", "on", "or", "our", "should", "so", "that",
    "the", "their", "this", "through", "to", "we", "will", "with", "within", "all",
    "any", "not", "no", "also", "both", "each", "supplier", "solution", "platform",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop common words, strip a plural 's'."""
    words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    tokens = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s"):
            word = word[:-1]
        tokens.append(word)
    return tokens


@lru_cache(maxsize=1)
def load_library() -> list[dict]:
    with open(config.CAPABILITY_LIBRARY, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _bm25_index() -> BM25Okapi:
    docs = [tokenize(f"{c['category']} {c['title']} {c['statement']} {c['limits']}") for c in load_library()]
    return BM25Okapi(docs)


def search(query: str, top_k: int = 4) -> list[dict]:
    """Return the top_k capabilities for a query, each with a 'score' added."""
    library = load_library()
    scores = _bm25_index().get_scores(tokenize(query))
    ranked = sorted(zip(library, scores), key=lambda pair: pair[1], reverse=True)
    return [{**cap, "score": round(float(score), 2)} for cap, score in ranked[:top_k]]


def get_by_ids(ids: list[str]) -> list[dict]:
    by_id = {c["id"]: c for c in load_library()}
    return [by_id[i] for i in ids if i in by_id]
