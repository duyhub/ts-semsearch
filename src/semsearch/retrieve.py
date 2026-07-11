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
from .normalize import canonicalize, doc_tokens, expand_query

# BM25F-lite field weights (T1b ablation). A field contributes its text N times in the
# flat doc; rank_bm25 reads the repeats as raised term frequency — no new dependency, no
# custom BM25F scorer. All-1s reproduces the historical flat concat token-for-token.
#
# CHOSEN: attributes x2, everything else x1. The ablation OVERTURNED the intuitive
# "boost identity fields" hypothesis: boosting name/brand REGRESSED official tune NDCG@5
# (0.9592 -> 0.942, driven by proper-name token collisions — e.g. P009 "cây xăng ... hạ
# long" where a name-matching POI outranks the gold gas station). The full pipeline already
# resolves identity via dense + subject-corroboration, so the lexical headroom is in the
# INTENT fields, not identity. Among intent fields, category boosting carried its own
# collision regression (P009 -0.044, only masked in aggregate) and a lone tags boost hurt
# synth150; attributes x2 was the one variant that improved MONOTONICALLY — tune@111 0.9608
# (+0.0016, zero per-query regressions), tune@1000 NDCG +0.0213, synth150 NDCG +0.0024.
# Attributes are the sponsor's controlled constraint taxonomy (10 canonical need-terms), so
# boosting them over free-text description/address is the principled, lowest-risk upgrade.
#
# district/city are pinned at 1 (NEVER 0): dropping them to stop the 'trà' ~ 'Sơn Trà'
# fold-collision previously regressed tune NDCG@5 0.959 -> 0.948 (location overlap genuinely
# helps the eval). The 'trà sữa' -> café fix is carried purely by the parser drink-category
# curation (CATEGORY_KEYWORDS), honored by the category hard-filter — not by starving the
# lexical location signal.
FIELD_WEIGHTS: dict[str, int] = {
    "name": 1, "brand": 1, "category": 1, "sub_category": 1,
    "tags": 1, "attributes": 2, "district": 1, "city": 1,
    "address": 1, "description": 1,
}

# Query tokens shorter than this are never fuzzy-canonicalized against the lexicon
# (mirrors normalize.canonicalize's own min_len — short tokens collide too easily).
_OOV_MIN_LEN = 4


def lexical_doc(p: POI) -> str:
    """Field-weighted BM25 doc (SPEC §5): identity/intent fields repeated per FIELD_WEIGHTS.

    A field contributes its text `weight` times; folding+tokenization downstream turns the
    repeats into raised token frequencies (BM25F-lite). Empty optional fields (brand,
    sub_category, blank description) contribute nothing — token-identical to the old flat
    concat when every weight is 1.
    """
    fields = {
        "name": p.name,
        "brand": p.brand or "",
        "category": p.category,
        "sub_category": p.sub_category or "",
        "district": p.district,
        "city": p.city,
        "address": p.address,
        "attributes": " ".join(p.attributes),
        "tags": " ".join(p.tags),
        "description": p.description,
    }
    parts: list[str] = []
    for name, text in fields.items():
        if text:
            parts.extend([text] * FIELD_WEIGHTS.get(name, 1))
    return " ".join(parts)


class BM25Index:
    def __init__(self, pois: Sequence[POI]):
        self.poi_ids = [p.poi_id for p in pois]
        corpus = [doc_tokens(lexical_doc(p)) for p in pois]
        self.bm25 = BM25Okapi(corpus)
        # Term lexicon = every token across the (field-weighted) docs. Repetition does
        # not change the token SET, so the lexicon is identical regardless of FIELD_WEIGHTS.
        # Drives OOV typo canonicalization in search(): an out-of-vocab query token is
        # snapped onto its unique edit-1 lexicon neighbour before scoring, giving typo'd
        # queries lexical recall in every mode with no LLM.
        self.lexicon: set[str] = set()
        for toks in corpus:
            self.lexicon.update(toks)

    def _canonicalize_oov(self, q_tokens: list[str]) -> list[str]:
        """Snap out-of-vocab query tokens onto their unique edit-1 lexicon neighbour.

        Delegates to normalize.canonicalize (exact match -> unchanged; len < min_len ->
        untouched; ambiguous edit-1 -> refused/untouched; unique edit-1 -> substituted),
        so in-vocab and short tokens pass through byte-identically — a clean, fully
        in-vocab query scores exactly as it would with no canonicalization at all.
        """
        return [canonicalize(t, self.lexicon, min_len=_OOV_MIN_LEN) or t for t in q_tokens]

    def search(self, query_text: str, k: int | None = None, *,
               drop: set[str] | None = None) -> list[tuple[str, float]]:
        """Score the FULL corpus (no top-k cut, OV1); return ranked (poi_id, score).

        `drop` removes tokens (a lifted district reference, e.g. {"quan", "1"})
        from the query before scoring, so they stop inflating every POI that
        carries them in a field — the quán/quận fold-collision where a District-N
        query token matches every District-N POI's district. Never emptied: if
        dropping would remove all tokens, the full query is kept.

        OOV typo tokens are canonicalized against the index lexicon AFTER expansion
        (so 'q1'->'quan 1' etc. resolve first) and BEFORE the drop de-pollution.
        """
        q_tokens = self._canonicalize_oov(expand_query(query_text))
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
