# Retrieval & Ranking Methodology

How the engine turns a Vietnamese need-based query into ranked POIs with
explanations, and how we measure it. Numbers below are reproduced by
`scripts/report_metrics.py` into [`reports/metrics.md`](../reports/metrics.md).

## Pipeline

```
query ─▶ normalize (fold diacritics, expand abbreviations/slang, q-numbers)
      ─▶ parse intent (category, required attributes, location anchor, time)   [rule-based]
      ─▶ retrieve: BM25 (folded tokens) + dense (bge-m3) ──RRF──▶ hybrid relevance
      ─▶ re-rank ALL POIs by 7 interpretable signals (semantic = hybrid relevance)
      ─▶ explanations (reasons derived only from true signal values)
```

Retrieval scores the **full 111-POI corpus** (no top-k truncation); the ranker
re-orders all of it. Attributes/category are **soft signals, not hard filters** —
an earlier filter-then-rank design lowered recall by deleting relevant POIs
(ablation: Recall@5 0.954→0.879), so it was removed. Because the `semantic`
signal *is* the hybrid relevance, all-weight-on-semantic reproduces hybrid, so
the tuned re-ranker is `full ≥ hybrid` by construction (ablation confirms it).

## The 7 signals — 1:1 with the sponsor's `Ranking_Signals`

| Sponsor signal | Our signal | Definition |
|---|---|---|
| `relevance_score` | **semantic** | hybrid BM25+dense RRF relevance, calibrated to [0,1] by a fixed max (not per-query min-max) |
| `distance_score` | **distance** | `exp(-d_km/3)` from the resolved anchor; neutral 0.5 without one |
| `rating_score` | **rating** | Bayesian `(v/(v+m))·R + (m/(v+m))·C`, low prior `m` so the narrow 3.8–4.7 band still varies |
| `popularity_score` | **popularity** | `popularity/100` |
| `review_signal` | **review** | query need-terms matched against POI tags + description (distinct from structured attributes) |
| `business_attributes` | **attributes** + **open_now** | taxonomy attribute match, plus the time dimension: open at the (injected) query time, handling `24/7` and overnight hours |
| `freshness_score` | — (documented) | **not implementable** on this dataset (no recency field); a production roadmap item |

Six sponsor signals are implemented (business_attributes splits into the
structured-attribute match and the open-now time check); `freshness` is
explicitly accounted for rather than faked. Every result keeps a per-signal
breakdown, which powers both the UI bars and the explanations.

## Explanations (faithful by construction)

Each result carries 1–4 Vietnamese reasons generated **only** from verifiable
values — matched attributes (`✓ wifi, ✓ yên tĩnh`), distance to the anchor,
rating + review count, opening hours. A faithfulness validator re-checks every
produced string against the POI and rejects any attribute or number not backed
by the data, so a hallucinated reason cannot ship (`tests/test_explain.py`).

## Evaluation protocol

- **Split:** the 60 labeled queries are split **40 tune / 20 test**, stratified
  by difficulty (Hard/Medium/Easy → 8/10/2 in test), fixed seed, committed to
  `data/eval_split.json`.
- **Integrity (NFR-6):** the test split never influences code, weights, or
  vocabularies. Weights are tuned on **tune only** via regularized coordinate
  ascent (coarse grid, minimum-improvement margin, 0.05 floor so all 7 signals
  stay live). Test is evaluated **once per milestone**. A test
  (`tests/test_integrity.py`) asserts no eval-query→POI mapping is hardcoded in
  `src/` and that the splits never overlap.
- **Metrics:** Recall@3/5, NDCG@5 (graded gains 3/2/1), MRR, reported overall
  and **per-difficulty and per-query-category with n**, plus a bootstrap CI so a
  20-query test number is never a bare point estimate.

## Results (see `reports/metrics.md`)

| Gate | Result |
|---|---|
| G1 BM25 Recall@5 (tune) | 0.917 (≥0.55) |
| G2 hybrid > max(bm25,dense) NDCG@5 (tune) | 0.922 > 0.881 |
| G3 full NDCG@5 / Recall@3 (**test**) | **0.884 / 0.933** (≥0.80 / ≥0.75) |
| G4 warm p95 latency | 2 ms (<200 ms) |
| G5 robustness | 138/138 checks pass |

**Ablation (tune):** random 0.005 → BM25 0.861 → dense 0.881 → hybrid 0.922 →
full re-rank **0.935** (NDCG@5) — each stage adds value.

## Honest caveats

- BM25 is unusually strong here because the **synthetic** POI descriptions embed
  the intent words literally; the hybrid/re-rank lift is therefore modest on the
  public set and concentrated in **Discovery** and **no-category-word intent**
  queries, where the need isn't stated in the text.
- The 20-query test split is small; we report bootstrap CIs (e.g. NDCG@5 0.884,
  95% CI [0.786–0.962]) and per-cell n rather than a bare headline. `POI Search`
  (n=1) and `Category Search` (n=2) cells are anecdotal.

## Models & provider posture

Local **`BAAI/bge-m3`** (multilingual embeddings) is the primary provider — the
build, tuning, and gates all run against it, so the demo has no hard network
dependency (NFR-3). Amazon Bedrock (`cohere.embed-multilingual-v3` / Titan v2,
Claude for LLM parsing) is a **selectable, measured** provider for
Built-with-AWS eligibility, never the default. Provider comparison lives in
`reports/embedding-choice.md`; caches are provider-stamped so a switch can never
silently mix vector spaces.
