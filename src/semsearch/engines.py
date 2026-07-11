"""Ranking engines — the MEASUREMENT factories behind run_eval/report_metrics/tune.

An engine is a RankFn: EvalQuery -> relevance-ordered list[poi_id].

EVAL INTEGRITY: every factory here is local-by-definition. `make_full_ranker` pins BOTH
`provider='local'` (embeddings) and `mode='local'` (LLM-parse default) so a deployment-mode
switch (SEMSEARCH_MODE / config.DEFAULT_MODE) can never leak network calls into a measured
number; the dense/hybrid factories construct raw indexes (no FullPipeline), so modes cannot
reach them by construction. Guarded by tests/test_integrity.py.
"""
from __future__ import annotations

import random
from typing import Sequence

from .data import POI
from .eval import RankFn
from .retrieve import BM25Index, DenseIndex, rrf_fuse


def make_random_ranker(pois: Sequence[POI], *, seed: int = 0) -> RankFn:
    """Deterministic random ranking — shuffles the full id universe per query.

    Seeded by query_id so runs are reproducible (NFR-5). Produces the near-zero
    metric floor the eval harness is validated against (Phase 1 gate).
    """
    ids = [p.poi_id for p in pois]

    def rank(q) -> list[str]:
        shuffled = list(ids)
        random.Random(f"{seed}:{q.query_id}").shuffle(shuffled)
        return shuffled

    return rank


def make_bm25_ranker(pois: Sequence[POI]) -> RankFn:
    """BM25 baseline over folded tokens (Phase 2, gate G1)."""
    index = BM25Index(pois)

    def rank(q) -> list[str]:
        return index.rank_ids(q.input_query)

    return rank


def make_dense_ranker(pois: Sequence[POI], provider: str = "local") -> RankFn:
    """Dense (embedding) retrieval baseline (Phase 3)."""
    from .embeddings import get_embedder

    index = DenseIndex(pois, get_embedder(provider))

    def rank(q) -> list[str]:
        return index.rank_ids(q.input_query)

    return rank


def make_hybrid_ranker(pois: Sequence[POI], provider: str = "local") -> RankFn:
    """BM25 + dense fused with RRF (Phase 3, gate G2)."""
    from .embeddings import get_embedder

    bm25 = BM25Index(pois)
    dense = DenseIndex(pois, get_embedder(provider))

    def rank(q) -> list[str]:
        fused = rrf_fuse([bm25.rank_ids(q.input_query), dense.rank_ids(q.input_query)])
        return [pid for pid, _ in fused]

    return rank


def make_full_ranker(pois: Sequence[POI], *, weights=None, now=None, provider: str = "local") -> RankFn:
    """Full pipeline: parse -> filter -> relax -> 9-signal re-rank (Phase 4, gate G3).

    Measurement factory: mode='local' is pinned so the LLM-parse default can never turn on
    under SEMSEARCH_MODE=cloud (provider pins embeddings; mode pins the rest)."""
    from .pipeline import FullPipeline
    from .rank import load_weights

    # weights is None -> use the tuned weights; an explicit {} is honored, not swapped (C23).
    pipe = FullPipeline(pois, weights=weights if weights is not None else load_weights(),
                        now=now, provider=provider, mode="local")

    def rank(q) -> list[str]:
        return pipe.rank_ids(q.input_query)

    return rank
