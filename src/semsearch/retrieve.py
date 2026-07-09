"""Lexical retrieval (SPEC §5). Phase 2: BM25 over folded tokens.

Dense + RRF fusion + full-corpus filtering arrive in later phases; the OV1
decision (filter the full corpus, no top-k truncation) is honored here by
scoring every POI and never pre-cutting the candidate set.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .data import POI
from .embeddings import Embedder, build_doc_matrix, embed_query, get_embedder, load_doc_matrix
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

    def search(self, query_text: str, k: int | None = None, *,
               drop: set[str] | None = None) -> list[tuple[str, float]]:
        """Score the FULL corpus (no top-k cut, OV1); return ranked (poi_id, score).

        `drop` removes tokens (a lifted district reference, e.g. {"quan", "1"})
        from the query before scoring, so they stop inflating every POI that
        carries them in a field — the quán/quận fold-collision where a District-N
        query token matches every District-N POI's district. Never emptied: if
        dropping would remove all tokens, the full query is kept.
        """
        q_tokens = expand_query(query_text)
        if drop:
            filtered = [t for t in q_tokens if t not in drop]
            if filtered:
                q_tokens = filtered
        scores = self.bm25.get_scores(q_tokens)
        order = sorted(range(len(self.poi_ids)), key=lambda i: scores[i], reverse=True)
        if k is not None:
            order = order[:k]
        return [(self.poi_ids[i], float(scores[i])) for i in order]

    def rank_ids(self, query_text: str) -> list[str]:
        return [pid for pid, _ in self.search(query_text)]


class DenseIndex:
    """Cosine retrieval over a provider-stamped embedding matrix (SPEC §4-5).

    Loads the cached matrix if present (asserting provider/model/POI-order match,
    A2), else builds and caches it. Vectors are L2-normalized, so cosine is one
    matvec.
    """

    def __init__(self, pois: Sequence[POI], embedder: Embedder | None = None):
        self.emb = embedder or get_embedder("local")
        self.poi_ids = [p.poi_id for p in pois]
        try:
            self.matrix = load_doc_matrix(self.emb, self.poi_ids)
        except FileNotFoundError:
            self.matrix = build_doc_matrix(pois, self.emb)

    def search(self, query_text: str, k: int | None = None) -> list[tuple[str, float]]:
        q = embed_query(self.emb, query_text)  # (d,), normalized
        sims = self.matrix @ q  # cosine (both L2-normalized)
        order = np.argsort(-sims)
        if k is not None:
            order = order[:k]
        return [(self.poi_ids[i], float(sims[i])) for i in order]

    def rank_ids(self, query_text: str) -> list[str]:
        return [pid for pid, _ in self.search(query_text)]


def rrf_fuse(rankings: Sequence[Sequence[str]], *, c: int = 60) -> list[tuple[str, float]]:
    """Reciprocal-rank fusion (SPEC §5). Combines ranked id lists into one fused
    ranking; the fused score also feeds the `semantic` ranking signal later.
    score(d) = sum_r 1 / (c + rank_r(d))   (rank 1-indexed).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] += 1.0 / (c + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
