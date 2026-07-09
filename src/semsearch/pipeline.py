"""Full search pipeline (SPEC §5-6), reworked per the G3 review.

Design: hybrid retrieval (BM25 + dense, RRF-fused over the FULL corpus, OV1)
provides the relevance backbone; the 7-signal linear ranker RE-ORDERS the whole
corpus using that hybrid relevance as its `semantic` signal plus attributes /
distance / rating / popularity / open_now / review. No destructive hard
filtering — attributes and category flow through the signals, not an AND-filter
that deletes recall. Because semantic == hybrid relevance, all-weight-on-semantic
reproduces hybrid exactly, so tuning makes full >= hybrid by construction.

C1: ranking the full corpus always returns a non-empty result for a valid query;
the API layer applies the top-N + popularity backstop for out-of-vocab inputs.
"""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from .data import POI, QueryIntent, RankedResult
from .embeddings import get_embedder
from .explain import generate_reasons
from .geo import Gazetteer
from .normalize import fold
from .parse import Parser
from .rank import DEFAULT_EVAL_NOW, DEFAULT_WEIGHTS, LinearRanker
from .retrieve import BM25Index, DenseIndex, rrf_fuse

RRF_C = 60
RRF_MAX = 2.0 / (RRF_C + 1)  # best possible fused score (rank 1 in both lists)


def _attrs_folded(p: POI) -> set[str]:
    return {fold(a) for a in p.attributes}


def _review_tokens(p: POI) -> set[str]:
    return set(fold(" ".join(p.tags) + " " + p.description).split())


class FullPipeline:
    def __init__(self, pois: Sequence[POI], *, weights: dict[str, float] | None = None,
                 now: datetime | None = None, provider: str = "local"):
        self.pois = list(pois)
        self.by_id = {p.poi_id: p for p in self.pois}
        self.dense = DenseIndex(self.pois, get_embedder(provider))
        self.bm25 = BM25Index(self.pois)
        self.gazetteer = Gazetteer(self.pois)
        self.parser = Parser(self.pois, self.gazetteer)
        C = sum(p.rating for p in self.pois) / len(self.pois)
        self.ranker = LinearRanker(weights or DEFAULT_WEIGHTS, now or DEFAULT_EVAL_NOW, C)
        self._attrs = {p.poi_id: _attrs_folded(p) for p in self.pois}
        self._review = {p.poi_id: _review_tokens(p) for p in self.pois}

    def _relevance(self, query_text: str) -> dict[str, float]:
        """Hybrid RRF relevance per POI, calibrated to [0,1] by a FIXED max (OV6:
        not per-query min-max)."""
        bm25_ids = self.bm25.rank_ids(query_text)
        dense_ids = [pid for pid, _ in self.dense.search(query_text)]
        fused = rrf_fuse([bm25_ids, dense_ids], c=RRF_C)
        return {pid: min(1.0, score / RRF_MAX) for pid, score in fused}

    def rank_scored(self, query_text: str) -> list[tuple[str, float, dict[str, float]]]:
        """Full-corpus ranking with per-signal breakdowns (used by the API/explanations)."""
        intent = self.parser.parse(query_text)
        rel = self._relevance(query_text)
        out: list[tuple[str, float, dict[str, float]]] = []
        for p in self.pois:
            s, b = self.ranker.score(
                rel.get(p.poi_id, 0.0), intent, p, self._attrs[p.poi_id], self._review[p.poi_id]
            )
            out.append((p.poi_id, s, b))
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def rank_ids(self, query_text: str) -> list[str]:
        return [pid for pid, _, _ in self.rank_scored(query_text)]

    def search(self, query_text: str, k: int = 10) -> tuple[QueryIntent, list[RankedResult]]:
        """Top-k results with per-signal breakdown + Vietnamese reasons (API/UI, FR-8)."""
        intent = self.parser.parse(query_text)
        results: list[RankedResult] = []
        for pid, score, breakdown in self.rank_scored(query_text)[:k]:
            poi = self.by_id[pid]
            results.append(
                RankedResult(poi=poi, score=score, breakdown=breakdown,
                             reasons=generate_reasons(intent, poi))
            )
        return intent, results
