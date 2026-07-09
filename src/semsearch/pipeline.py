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

from .data import POI, QueryIntent, RankedResult, content_tokens
from .embeddings import get_embedder
from .explain import generate_reasons
from .geo import Gazetteer, haversine
from .normalize import fold

# When a query resolves an explicit location anchor, "gần X" must mean near X:
# near-anchor POIs rank first, far ones drop to the tail (recall preserved).
ANCHOR_RADII_KM = (30.0, 150.0)  # try tight metro radius, then wider; else no gate
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
        self._content = {p.poi_id: content_tokens(p) for p in self.pois}  # subject-filter tokens

    def _relevance(self, query_text: str, intent: QueryIntent) -> dict[str, float]:
        """Hybrid RRF relevance per POI, calibrated to [0,1] by a FIXED max (OV6:
        not per-query min-max). A lifted district reference is stripped from the
        BM25 query (quán/quận de-pollution) — location is carried by the distance
        signal, not lexical token overlap. The dense side keeps the full query
        (embeddings don't token-double-count)."""
        drop = set(fold(intent.district).split()) if intent.district else None
        bm25_ids = [pid for pid, _ in self.bm25.search(query_text, drop=drop)]
        dense_ids = [pid for pid, _ in self.dense.search(query_text)]
        fused = rrf_fuse([bm25_ids, dense_ids], c=RRF_C)
        return {pid: min(1.0, score / RRF_MAX) for pid, score in fused}

    def rank_scored(self, query_text: str) -> list[tuple[str, float, dict[str, float]]]:
        """Full-corpus ranking with per-signal breakdowns (used by the API/explanations)."""
        intent = self.parser.parse(query_text)
        rel = self._relevance(query_text, intent)
        out: list[tuple[str, float, dict[str, float]]] = []
        for p in self.pois:
            s, b = self.ranker.score(
                rel.get(p.poi_id, 0.0), intent, p, self._attrs[p.poi_id], self._review[p.poi_id]
            )
            out.append((p.poi_id, s, b))
        out.sort(key=lambda t: t[1], reverse=True)
        out = self._constraint_filter(out, intent)
        if intent.anchor is not None:
            out = self._anchor_gate(out, intent)
        return out

    def _constraint_filter(self, ranked, intent):
        """Hard-filter to satisfy the query's expressed constraints (SPEC §6):
        location (district/city), subject (distinctive content terms), or category
        (only when the parse is fully explained — `has_residual` is False, which
        guards mis-parses like P019/P055). Returns MATCHES ONLY (may be fewer than
        the limit); relaxes the most-specific constraint first until non-empty (G5)."""
        filters = []
        if intent.district or intent.city:
            d, c = intent.district, intent.city
            filters.append(lambda pid: (d is None or self.by_id[pid].district == d)
                           and (c is None or self.by_id[pid].city == c))
        if intent.content_terms:
            terms = set(intent.content_terms)
            filters.append(lambda pid: terms <= self._content[pid])
        elif intent.category and not intent.has_residual:
            cat = intent.category
            filters.append(lambda pid: self.by_id[pid].category == cat)

        active = filters
        while active:
            keep = [t for t in ranked if all(f(t[0]) for f in active)]
            if keep:
                return keep
            active = active[:-1]  # relax subject/category first, then location
        return ranked

    def _anchor_gate(self, ranked, intent):
        """Float near-anchor POIs to the top; far ones become the tail. Relax the
        radius if too few survive; if even the widest radius yields <3, skip the
        gate (keep pure score order) rather than starve the result."""
        a = intent.anchor
        def near(radius):
            return [t for t in ranked if haversine(a.lat, a.lon, self.by_id[t[0]].lat,
                                                    self.by_id[t[0]].lon) <= radius]
        for radius in ANCHOR_RADII_KM:
            hits = near(radius)
            if len(hits) >= 3:
                keep = {t[0] for t in hits}
                far = [t for t in ranked if t[0] not in keep]  # both keep score order
                return hits + far
        return ranked

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
