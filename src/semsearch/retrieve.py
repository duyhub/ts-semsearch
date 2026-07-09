"""Lexical retrieval (SPEC §5). Phase 2: BM25 over folded tokens.

Dense + RRF fusion + full-corpus filtering arrive in later phases; the OV1
decision (filter the full corpus, no top-k truncation) is honored here by
scoring every POI and never pre-cutting the candidate set.
"""
from __future__ import annotations

from typing import Sequence

from rank_bm25 import BM25Okapi

from .data import POI
from .normalize import doc_tokens, expand_query


def lexical_doc(p: POI) -> str:
    """Fields BM25 indexes (SPEC §5): name/brand/category/address/attributes/tags + context."""
    parts = [
        p.name,
        p.brand or "",
        p.category,
        p.sub_category or "",
        p.district,
        p.city,
        p.address,
        " ".join(p.attributes),
        " ".join(p.tags),
        p.description,
    ]
    return " ".join(parts)


class BM25Index:
    def __init__(self, pois: Sequence[POI]):
        self.poi_ids = [p.poi_id for p in pois]
        corpus = [doc_tokens(lexical_doc(p)) for p in pois]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query_text: str, k: int | None = None) -> list[tuple[str, float]]:
        """Score the FULL corpus (no top-k cut, OV1); return ranked (poi_id, score)."""
        q_tokens = expand_query(query_text)
        scores = self.bm25.get_scores(q_tokens)
        order = sorted(range(len(self.poi_ids)), key=lambda i: scores[i], reverse=True)
        if k is not None:
            order = order[:k]
        return [(self.poi_ids[i], float(scores[i])) for i in order]

    def rank_ids(self, query_text: str) -> list[str]:
        return [pid for pid, _ in self.search(query_text)]
