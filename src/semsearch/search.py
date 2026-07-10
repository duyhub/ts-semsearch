"""SearchEngine facade (SPEC §2, §9). The frozen interface the API and UI build
against; later phases extend it with dense retrieval, the 9-signal ranker,
explanations, and an injected `now` clock (A1) — without changing the signature.

Phase 2: BM25-only ranking.
"""
from __future__ import annotations

from typing import Sequence

from .data import POI
from .retrieve import BM25Index


class SearchEngine:
    def __init__(self, pois: Sequence[POI]):
        self.pois = list(pois)
        self.by_id = {p.poi_id: p for p in self.pois}
        self.bm25 = BM25Index(self.pois)

    def search_ids(self, query_text: str) -> list[str]:
        """Return POI ids in ranked order. Phase 2 = BM25 only."""
        return self.bm25.rank_ids(query_text)
