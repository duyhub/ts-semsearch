"""Full search pipeline (SPEC §5-6), reworked per the G3 review.

Design: hybrid retrieval (BM25 + dense, RRF-fused over the FULL corpus, OV1)
provides the relevance backbone; the 9-signal linear ranker RE-ORDERS the whole
corpus using that hybrid relevance as its `semantic` signal plus attributes /
distance / rating / popularity / open_now / review. No destructive hard
filtering — attributes and category flow through the signals, not an AND-filter
that deletes recall. Because semantic == hybrid relevance, all-weight-on-semantic
reproduces hybrid exactly, so tuning makes full >= hybrid by construction.

C1: ranking the full corpus always returns a non-empty result for a valid query;
the API layer applies the top-N + popularity backstop for out-of-vocab inputs.
"""
from __future__ import annotations

import time
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

# A parser "distinctive subject" term (rare in POI names, df<=2) may hard-filter the
# results ONLY if the DENSE retriever corroborates it — a POI matching the term sits
# in the dense top-K. BM25 ranks a coincidental high-IDF proper-name token (e.g. "nhat"
# in "Thống Nhất") at #1, but dense understands the query and ranks it far down; this
# gate drops those spurious subjects. Structural constant (top ~9% of 111), NOT tuned
# on eval: genuine subjects sit at dense rank 1 vs spurious 45+, so any K in [5,30] is
# equivalent. See docs/superpowers/specs/2026-07-10-subject-filter-corroboration-design.md.
DENSE_SUBJECT_TOPK = 10


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

    def _relevance(self, query_text: str, intent: QueryIntent,
                   dense_ids: list[str], trace: dict | None = None) -> dict[str, float]:
        """Hybrid RRF relevance per POI, calibrated to [0,1] by a FIXED max (OV6:
        not per-query min-max). A lifted district reference is stripped from the
        BM25 query (quán/quận de-pollution) — location is carried by the distance
        signal, not lexical token overlap. The dense side keeps the full query
        (embeddings don't token-double-count); `dense_ids` is precomputed once by
        the caller so corroboration and fusion share the single dense pass."""
        drop = set(fold(intent.district).split()) if intent.district else None
        bm25_ids = [pid for pid, _ in self.bm25.search(query_text, drop=drop)]
        if trace is not None:
            trace["bm25Top"] = bm25_ids[:5]
        fused = rrf_fuse([bm25_ids, dense_ids], c=RRF_C)
        return {pid: min(1.0, score / RRF_MAX) for pid, score in fused}

    def rank_scored(self, query_text: str, trace: dict | None = None
                    ) -> list[tuple[str, float, dict[str, float]]]:
        """Full-corpus ranking with per-signal breakdowns (used by the API/explanations).

        When `trace` is a dict, it is populated with a deterministic, structural summary of
        the pipeline's decisions (retrieval tops, which constraints engaged/relaxed, whether
        the anchor gate fired) PLUS a `steps` list timing each stage — the per-request
        execution trace the /admin view renders. Timing wraps whole stages (never per-POI),
        so the cost is a handful of perf_counter() calls; when `trace` is None the path is
        byte-for-byte unchanged (the contract /v1/search never pays for it)."""
        steps = trace.setdefault("steps", []) if trace is not None else None

        def _timed(name, fn):
            """Run fn(), and if tracing, record {name, ms} for this stage."""
            if steps is None:
                return fn()
            t0 = time.perf_counter()
            result = fn()
            steps.append({"name": name, "ms": round((time.perf_counter() - t0) * 1000, 3)})
            return result

        intent = _timed("parse", lambda: self.parser.parse(query_text))
        dense_ids = _timed("dense_retrieval",
                           lambda: [pid for pid, _ in self.dense.search(query_text)])
        rel = _timed("lexical_fusion",
                     lambda: self._relevance(query_text, intent, dense_ids, trace=trace))

        def _score_all():
            out: list[tuple[str, float, dict[str, float]]] = []
            for p in self.pois:
                s, b = self.ranker.score(
                    rel.get(p.poi_id, 0.0), intent, p, self._attrs[p.poi_id], self._review[p.poi_id]
                )
                out.append((p.poi_id, s, b))
            out.sort(key=lambda t: t[1], reverse=True)
            return out

        out = _timed("rank_signals", _score_all)
        out = _timed("constraint_filter",
                     lambda: self._constraint_filter(out, intent, dense_ids, trace=trace))
        if intent.anchor is not None:
            gated = _timed("anchor_gate", lambda: self._anchor_gate(out, intent))
            if trace is not None:
                trace["anchorGateFired"] = gated is not out  # a new list ⇒ gate reordered
            out = gated
        elif trace is not None:
            trace["anchorGateFired"] = False
        if trace is not None:
            trace["denseTop"] = dense_ids[:5]
        return out

    def _corroborated_subjects(self, intent: QueryIntent, dense_ids: list[str]) -> set[str]:
        """Keep only the parser's distinctive `content_terms` that the DENSE retriever
        corroborates as central to the query — some POI whose folded name/text contains
        the term appears in the dense top-K. Filters out coincidental high-IDF proper-
        name collisions (BM25 ranks them #1; dense does not)."""
        dense_top = dense_ids[:DENSE_SUBJECT_TOPK]
        return {t for t in intent.content_terms
                if any(t in self._content[pid] for pid in dense_top)}

    def _constraint_filter(self, ranked, intent, dense_ids, trace: dict | None = None):
        """Hard-filter to satisfy the query's expressed constraints (SPEC §6):
        location (district/city), subject (distinctive content terms), or category
        (only when the parse is fully explained). Returns MATCHES ONLY (may be fewer
        than the limit); relaxes the most-specific constraint first until non-empty (G5).

        The subject filter fires only for DENSE-corroborated terms; a distinctive term
        the dense retriever discredits (a coincidental proper-name collision like "nhat"
        in "Thống Nhất") is dropped, and — since it never described a real subject — it
        no longer blocks the category filter either. Genuine unexplained content (e.g.
        P055's "mua/sắm") still blocks category, preserving the mis-parse guard."""
        subject_terms = self._corroborated_subjects(intent, dense_ids)
        discredited = set(intent.content_terms) - subject_terms  # spurious distinctive terms
        meaningful_residual = [t for t in intent.residual_terms if t not in discredited]

        filters = []
        engaged: list[str] = []  # human-readable labels for the /admin trace
        if intent.district or intent.city:
            d, c = intent.district, intent.city
            filters.append(lambda pid: (d is None or self.by_id[pid].district == d)
                           and (c is None or self.by_id[pid].city == c))
            engaged.append("location")
        if subject_terms:
            filters.append(lambda pid: subject_terms <= self._content[pid])
            engaged.append("subject")
        elif intent.category and not meaningful_residual:
            cat = intent.category
            filters.append(lambda pid: self.by_id[pid].category == cat)
            engaged.append("category")

        def _trace(applied_labels: list[str]) -> None:
            if trace is not None:
                trace["constraintsEngaged"] = list(engaged)
                trace["constraintsApplied"] = applied_labels  # after relaxation
                trace["constraintRelaxed"] = len(applied_labels) < len(engaged)

        active = filters
        while active:
            keep = [t for t in ranked if all(f(t[0]) for f in active)]
            if keep:
                _trace(engaged[:len(active)])  # engaged labels align with filter order
                return keep
            active = active[:-1]  # relax subject/category first, then location
        _trace([])  # every constraint relaxed away; fell back to full ranking
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
