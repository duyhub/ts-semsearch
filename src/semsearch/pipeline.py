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

import logging
import os
import threading
from dataclasses import replace
from datetime import datetime
from typing import Sequence

from .config import VALID_MODES, resolve_mode, resolve_query_rewrite
from .data import POI, QueryIntent, RankedResult, content_tokens
from .embeddings import (
    BEDROCK_PROVIDERS,
    get_embedder,
    is_no_credentials_error,
    resolve_provider,
)
from .explain import generate_reasons
from .geo import Gazetteer, haversine
from .llm_parse import LLMParser, merge_intent
from .normalize import fold

logger = logging.getLogger(__name__)

# FR-4 / NFR-5: the LLM intent parse defaults OFF in local/local-first modes (deterministic
# /v1/search) and ON in cloud mode (remote hosting implies network). Explicit
# SEMSEARCH_LLM_PARSE always wins:
#   "bedrock" | "on" -> force ON (full resolution chain: Bedrock -> OpenAI)
#   "openai"         -> force ON, pin OpenAI directly (skip all Bedrock probes)
#   "off"            -> force OFF
#   anything else    -> WARNING + off (never silently on)
LLM_PARSE_ENV = "SEMSEARCH_LLM_PARSE"

# When a query resolves an explicit location anchor, "gần X" must mean near X:
# near-anchor POIs rank first, far ones drop to the tail (recall preserved).
ANCHOR_RADII_KM = (30.0, 150.0)  # try tight metro radius, then wider; else no gate
from .parse import Parser
from .rank import DEFAULT_EVAL_NOW, DEFAULT_WEIGHTS, LinearRanker
from .retrieve import BM25Index, DenseIndex, rrf_fuse

RRF_C = 60  # the fused-score calibration max is derived per query from the lists actually
# fused: k non-empty lists -> max k/(RRF_C+1) (rank 1 in each). A fixed 2/(c+1) would
# silently cap the semantic signal at 0.5 whenever dense is absent (BM25-only floor).

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
                 now: datetime | None = None, provider: str | None = None,
                 mode: str | None = None):
        """`mode` selects the deployment posture — see src/semsearch/config.py.

        Pinning contract (eval integrity): an EXPLICIT `mode=` argument FULLY pins the
        pipeline — resolve_mode() is never called, so neither SEMSEARCH_MODE nor an edited
        DEFAULT_MODE can reach it (an invalid explicit mode raises: programmer error). Only
        `mode=None` env-resolves. An EXPLICIT `provider=` is the embeddings expert override
        (skips mode resolution for embeddings); `mode` still governs the LLM-parse default,
        which is why measurement entry points pin BOTH (provider='local', mode='local')."""
        self.pois = list(pois)
        self.by_id = {p.poi_id: p for p in self.pois}
        if mode is not None and mode not in VALID_MODES:
            raise ValueError(f"invalid mode {mode!r} (valid: {', '.join(VALID_MODES)})")
        self.mode = mode or resolve_mode()
        self.dense: DenseIndex | None
        if provider is not None:
            # EXPERT OVERRIDE — exactly the pre-mode path. Coherent provider choice BEFORE
            # building the index: a bedrock provider whose preflight fails (no creds/model
            # access/timeout) degrades to local here, so the whole run stays in one vector
            # space (A2) and the demo never depends on the network.
            provider = resolve_provider(provider)
            try:
                self.dense = DenseIndex(self.pois, get_embedder(provider))
            except Exception as exc:  # noqa: BLE001 - bedrock-only construction fallback
                if provider not in BEDROCK_PROVIDERS:
                    raise  # a LOCAL build failure is a setup bug — propagate loudly
                # The preflight only pings one string; the network can still drop DURING
                # the 111-doc matrix build. Rebuild in the local space (one warning) so
                # construction keeps the 'coherent for the entire run' guarantee.
                logger.warning(
                    "Bedrock provider %r failed while building the doc matrix (%s: %s); "
                    "rebuilding with local bge-m3.",
                    provider, type(exc).__name__, exc,
                )
                self.dense = DenseIndex(self.pois, get_embedder("local"))
        else:
            self.dense = self._dense_for_mode(self.mode)
        self.bm25 = BM25Index(self.pois)
        self.gazetteer = Gazetteer(self.pois)
        self.parser = Parser(self.pois, self.gazetteer)
        C = sum(p.rating for p in self.pois) / len(self.pois)
        self.ranker = LinearRanker(weights or DEFAULT_WEIGHTS, now or DEFAULT_EVAL_NOW, C)
        self._attrs = {p.poi_id: _attrs_folded(p) for p in self.pois}
        self._review = {p.poi_id: _review_tokens(p) for p in self.pois}
        self._content = {p.poi_id: content_tokens(p) for p in self.pois}  # subject-filter tokens
        # FR-4 gate: cloud mode implies network, so the LLM parse defaults ON there; the
        # other modes keep today's deterministic default (OFF). Explicit env always wins.
        # `_llm_enabled_via` records HOW it was enabled, so the warn-once message tells a
        # cloud operator about the mode default instead of an env var they never set.
        llm_env = os.environ.get(LLM_PARSE_ENV)
        llm_on, prefer = False, "auto"
        if llm_env is None:
            llm_on = self.mode == "cloud"
            self._llm_enabled_via = "cloud mode default"
        elif llm_env in ("bedrock", "on"):
            llm_on = True
            self._llm_enabled_via = f"{LLM_PARSE_ENV}={llm_env}"
        elif llm_env == "openai":
            llm_on, prefer = True, "openai"  # pin OpenAI directly, skip Bedrock probes
            self._llm_enabled_via = f"{LLM_PARSE_ENV}=openai"
        elif llm_env == "off":
            self._llm_enabled_via = f"{LLM_PARSE_ENV}=off"
        else:
            logger.warning(
                "unknown %s value %r (valid: off, on, bedrock, openai); LLM parse stays off.",
                LLM_PARSE_ENV, llm_env,
            )
            self._llm_enabled_via = f"{LLM_PARSE_ENV}={llm_env} (unknown)"
        self._llm_parser = LLMParser(prefer=prefer) if llm_on else None
        self._llm_warned = False  # log the "LLM unavailable" warning at most once
        self._llm_warn_lock = threading.Lock()  # latch is check-then-set; API serves threaded
        # FR-4: when on, the LLM's corrected_query REPLACES the raw text for retrieval. It rides
        # the LLM parse (no extra call), so it is inert whenever the parse is off/unavailable.
        self._query_rewrite = resolve_query_rewrite()

    def _dense_for_mode(self, mode: str) -> DenseIndex | None:
        """Construction-time embeddings resolution per deployment mode — chosen ONCE,
        coherent for the whole run (A2), every degradation logged loudly.

        local:       today's behavior — local bge-m3; a failure raises (setup bug).
        local-first: probe local FIRST (loads bge-m3 + one embed, so a broken host fails
                     here, not mid-demo); on failure walk the cloud chain.
        cloud:       NEVER touch local (remote hosting without the 2.3GB model);
                     walk the cloud chain; all failing -> BM25-only floor (None)."""
        if mode == "local":
            return DenseIndex(self.pois, get_embedder("local"))
        if mode == "local-first":
            try:
                emb = get_embedder("local")
                emb.embed(["khởi động"])  # probe: load the model NOW, fail fast if broken
                return DenseIndex(self.pois, emb)
            except Exception as exc:  # noqa: BLE001 - degrade to cloud, LOUDLY
                logger.warning(
                    "mode=local-first: local embeddings unavailable (%s: %s); "
                    "walking the cloud chain.", type(exc).__name__, exc,
                )
        return self._cloud_dense()

    def _cloud_dense(self) -> DenseIndex | None:
        """Walk the cloud embedding providers (bedrock-cohere, then bedrock-titan — each
        already walks the region-fallback chain internally). First one whose probe + doc
        build succeeds wins; everything failing lands on the BM25-only floor (dense=None),
        loudly."""
        for prov in BEDROCK_PROVIDERS:
            try:
                emb = get_embedder(prov)
                # probe pins the provider's region (walks the chain) or raises
                emb.embed(["khởi động"], input_type="search_query")
                return DenseIndex(self.pois, emb)
            except Exception as exc:  # noqa: BLE001 - try the next cloud provider
                if is_no_credentials_error(exc):
                    # Tradeoff: missing credentials are ACCOUNT-wide — probing the other
                    # provider (x3 regions) can only burn more timeouts for the same answer,
                    # so short-circuit straight to the floor. Network/model errors keep the
                    # full provider walk (those CAN differ per provider/region).
                    logger.warning(
                        "cloud embeddings: no AWS credentials (%s) — skipping the remaining "
                        "cloud providers.", type(exc).__name__,
                    )
                    break
                logger.warning(
                    "cloud embeddings provider %r unavailable (%s: %s); trying the next.",
                    prov, type(exc).__name__, exc,
                )
        logger.warning(
            "no embeddings provider available in mode %r; running the BM25-only floor "
            "(keyword retrieval only — no dense semantic ranking).", self.mode,
        )
        return None

    def _relevance(self, query_text: str, intent: QueryIntent,
                   dense_ids: list[str]) -> dict[str, float]:
        """Retrieval relevance per POI, calibrated to [0,1] by a FIXED per-shape max (OV6:
        not per-query min-max): the best fused score is one rank-1 reciprocal vote per
        NON-EMPTY list, i.e. k/(RRF_C+1) — so BM25-only (floor mode, or a degraded query
        vector) calibrates against 1/(c+1) instead of silently capping at 0.5. A lifted
        district reference is stripped from the BM25 query (quán/quận de-pollution) —
        location is carried by the distance signal, not lexical token overlap. The dense
        side keeps the full query (embeddings don't token-double-count); `dense_ids` is
        precomputed once by the caller so corroboration and fusion share the single dense
        pass."""
        drop = set(fold(intent.district).split()) if intent.district else None
        bm25_ids = [pid for pid, _ in self.bm25.search(query_text, drop=drop)]
        lists = [bm25_ids, dense_ids] if self.dense is not None else [bm25_ids]
        fused = rrf_fuse(lists, c=RRF_C)
        rrf_max = max(1, sum(1 for l in lists if l)) / (RRF_C + 1)
        return {pid: min(1.0, score / rrf_max) for pid, score in fused}

    def resolve_intent(self, query_text: str) -> QueryIntent:
        """The ONE intent resolution for a query — public because the API layer must use
        the SAME intent object for ranking, the intent echo, and reasons[] (a rule-only
        re-parse there would contradict LLM-merged results). When the LLM gate is off
        (default) this is byte-identical to `self.parser.parse` — the gate short-circuits
        before any new code runs. When SEMSEARCH_LLM_PARSE=bedrock, ONE Claude parse runs on
        the RAW text; on ANY failure (network, creds, bad JSON) the rule intent is used alone,
        with a single warning logged once (lock: uvicorn serves on a threadpool).

        Query rewrite (FR-4): with the parse on AND `resolve_query_rewrite()` True, the parse's
        `corrected_query` (typos fixed, diacritics restored) REPLACES the raw text for the rule
        parse (and, downstream, BM25/dense/subject). `raw` on the returned intent always holds
        the ORIGINAL text — the API echo must never show the rewrite. The LLM fires at most
        once per query regardless."""
        if self._llm_parser is None:
            return self.parser.parse(query_text)
        llm_out = None
        try:
            llm_out = self._llm_parser.parse(query_text)  # RAW text; never raises; None on failure
        except Exception:  # noqa: BLE001 - defensive; the LLM parse must never break a query
            llm_out = None
        if llm_out is None:
            with self._llm_warn_lock:
                if not self._llm_warned:
                    logger.warning(
                        "LLM parse (enabled via %s) unavailable; serving rule-parsed "
                        "results. Run scripts/check_bedrock.py to diagnose.",
                        self._llm_enabled_via,
                    )
                    self._llm_warned = True
            return self.parser.parse(query_text)
        corrected = (llm_out.get("corrected_query") or None) if self._query_rewrite else None
        if corrected == query_text:  # extra safety; _validate already no-ops exact matches
            corrected = None
        rule_intent = self.parser.parse(corrected or query_text)  # parse the text we retrieve on
        merged = merge_intent(rule_intent, llm_out)
        return replace(merged, raw=query_text, corrected_query=corrected)  # raw = ORIGINAL

    def rank_scored(self, query_text: str, *, intent: QueryIntent | None = None,
                    ) -> list[tuple[str, float, dict[str, float]]]:
        """Full-corpus ranking with per-signal breakdowns (used by the API/explanations).
        `intent` may be passed in so a single query resolves its intent once (the LLM parse
        fires at most once per query — see `search`); when omitted it is resolved here."""
        if intent is None:
            intent = self.resolve_intent(query_text)
        # FR-4 query rewrite: retrieval runs on the LLM-corrected text when present (the rule
        # parse already did in resolve_intent); the raw text is display-only. dense + BM25 +
        # subject corroboration (fed by this single dense_ids pass) all see the corrected text.
        retrieval_text = intent.corrected_query or query_text
        # one dense pass, reused below; the BM25-only floor (dense=None) has no dense opinion
        dense_ids = [pid for pid, _ in self.dense.search(retrieval_text)] if self.dense else []
        rel = self._relevance(retrieval_text, intent, dense_ids)
        out: list[tuple[str, float, dict[str, float]]] = []
        for p in self.pois:
            s, b = self.ranker.score(
                rel.get(p.poi_id, 0.0), intent, p, self._attrs[p.poi_id], self._review[p.poi_id]
            )
            out.append((p.poi_id, s, b))
        out.sort(key=lambda t: t[1], reverse=True)
        out = self._constraint_filter(out, intent, dense_ids)
        if intent.anchor is not None:
            out = self._anchor_gate(out, intent)
        return out

    def _corroborated_subjects(self, intent: QueryIntent, dense_ids: list[str]) -> set[str]:
        """Keep only the parser's distinctive `content_terms` that the DENSE retriever
        corroborates as central to the query — some POI whose folded name/text contains
        the term appears in the dense top-K. Filters out coincidental high-IDF proper-
        name collisions (BM25 ranks them #1; dense does not)."""
        dense_top = dense_ids[:DENSE_SUBJECT_TOPK]
        return {t for t in intent.content_terms
                if any(t in self._content[pid] for pid in dense_top)}

    def _constraint_filter(self, ranked, intent, dense_ids):
        """Hard-filter to satisfy the query's expressed constraints (SPEC §6):
        location (district/city), subject (distinctive content terms), or category
        (only when the parse is fully explained). Returns MATCHES ONLY (may be fewer
        than the limit); relaxes the most-specific constraint first until non-empty (G5).

        The subject filter is ALL-OR-NOTHING (C7): it fires only when EVERY distinctive
        content term is DENSE-corroborated (some POI carrying it sits in the dense top-K).
        If only a subset corroborates, the whole subject filter is dropped — otherwise the
        surviving subset admits wrong POIs (e.g. 'hue' from an unrelated ATM in a
        'highlands coffee nguyễn huệ' search). When dropped, all content terms are treated
        as discredited so they no longer block the category filter either; genuine
        unexplained content (e.g. P055's "mua/sắm") still blocks category, preserving the
        mis-parse guard."""
        corroborated = self._corroborated_subjects(intent, dense_ids)
        if intent.content_terms and corroborated == set(intent.content_terms):
            subject_terms = corroborated
            discredited: set[str] = set()
        else:  # partial (or no) corroboration -> no subject filter; nothing blocks category
            subject_terms = set()
            discredited = set(intent.content_terms)
        meaningful_residual = [t for t in intent.residual_terms if t not in discredited]

        filters = []
        if intent.district or intent.city:
            d, c = intent.district, intent.city
            filters.append(lambda pid: (d is None or self.by_id[pid].district == d)
                           and (c is None or self.by_id[pid].city == c))
        if subject_terms:
            filters.append(lambda pid: subject_terms <= self._content[pid])
        elif intent.category and not meaningful_residual:
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
        intent = self.resolve_intent(query_text)  # resolved once; passed through to ranking
        results: list[RankedResult] = []
        for pid, score, breakdown in self.rank_scored(query_text, intent=intent)[:k]:
            poi = self.by_id[pid]
            results.append(
                RankedResult(poi=poi, score=score, breakdown=breakdown,
                             reasons=generate_reasons(intent, poi))
            )
        return intent, results
