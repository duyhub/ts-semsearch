"""Ranking engines. Phase 1 ships only the random baseline (the eval-harness
sanity check); BM25 / dense / hybrid / full arrive in later phases.

An engine is a RankFn: EvalQuery -> relevance-ordered list[poi_id].
"""
from __future__ import annotations

import random
from typing import Sequence

from .data import POI
from .eval import RankFn
from .retrieve import BM25Index


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
