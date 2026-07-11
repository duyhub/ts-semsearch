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
    """Fields BM25 indexes (SPEC §5): name/brand/category/address/attributes/tags + context.

    NB (Fix 3): district/city are kept here. Dropping them to stop the 'trà' ~ 'Sơn Trà'
    fold-collision regressed tune NDCG@5 0.959 -> 0.948 (location overlap genuinely helps
    the eval), so the 'trà sữa' -> café fix is carried purely by the parser drink-category
    curation (CATEGORY_KEYWORDS), which the category hard-filter then honors.
    """
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
    """Cosine retrieval over a provider+corpus-stamped embedding matrix (SPEC §4-5).

    Loads the cached matrix if present (asserting provider/model/POI-order match,
    A2), else builds and caches it. Vectors are L2-normalized, so cosine is one
    matvec.

    The cache path is namespaced by both provider/model AND a corpus fingerprint
    (embeddings._corpus_hash of the POI id sequence), so two different corpora sharing
    a provider/model — e.g. the official 111-POI set and a synthetic superset — never
    contend for the same file. load_doc_matrix can still raise the A2 ValueError if a
    stale/corrupted manifest somehow occupies the resolved path; we catch that alongside
    FileNotFoundError and rebuild rather than propagate it, because a fresh build is
    coherent by construction and never serves a mismatched matrix (A2's actual intent
    is refuse-to-SERVE-garbage, not refuse-to-rebuild).
    """

    def __init__(self, pois: Sequence[POI], embedder: Embedder | None = None):
        self.emb = embedder or get_embedder("local")
        self.poi_ids = [p.poi_id for p in pois]
        try:
            self.matrix = load_doc_matrix(self.emb, self.poi_ids)
        except (FileNotFoundError, ValueError):
            self.matrix = build_doc_matrix(pois, self.emb)

    def search(self, query_text: str, k: int | None = None) -> list[tuple[str, float]]:
        q = embed_query(self.emb, query_text)  # (d,), normalized
        if not np.any(q):
            # Degraded query embed (bedrock failure -> zero vector): dense has NO
            # opinion. An all-zero matvec argsorted would return DATASET-ORDER ids,
            # polluting RRF fusion and the subject-corroboration top-K. An empty
            # ranking makes fusion defer cleanly to BM25. (A real embedding of real
            # text is unit-norm, never all-zero.)
            return []
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
