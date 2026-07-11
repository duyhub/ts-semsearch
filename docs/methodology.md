# Retrieval & Ranking Methodology

How the engine turns a Vietnamese need-based query into ranked POIs with
explanations, and how we measure it. Numbers below are reproduced by
`scripts/report_metrics.py` into [`reports/metrics.md`](../reports/metrics.md).

## Pipeline

```
query ─▶ normalize (fold diacritics, expand abbreviations/slang, q-numbers)
      ─▶ parse intent (category, attributes, location anchor, subject, time)   [rule-based]
      ─▶ retrieve: BM25 (folded tokens) + dense (bge-m3) ──RRF──▶ hybrid relevance
      ─▶ re-rank ALL POIs by 9 interpretable signals (semantic = hybrid relevance)
      ─▶ hard-constraint filter, relaxing (pure category / location / subject)
      ─▶ explanations (reasons derived only from true signal values)
```

Retrieval scores the **full 111-POI corpus** (no top-k truncation); the ranker
re-orders all of it. Ranking is driven by **soft signals** — an early
filter-then-rank design that pre-deleted candidates lowered recall (ablation:
Recall@5 0.954→0.879), so filtering never *precedes* ranking. Because the
`semantic` signal *is* the hybrid relevance, all-weight-on-semantic reproduces
hybrid, so the tuned re-ranker is `full ≥ hybrid` by construction (ablation
confirms it).

**BM25 document composition.** BM25 scores a flat, folded-token document per POI
(name/brand/category/sub_category/district/city/address/attributes/tags/description),
with the `attributes` field weighted ×2 (token repetition — no custom BM25F
scorer). The finding behind that choice cuts against intuition and is reported
honestly: boosting *identity* fields (name/brand) instead **regressed** tune
NDCG@5, via proper-name token collisions — identity is already resolved by dense
retrieval + subject corroboration, so the lexical headroom sits in *intent*
fields. `attributes` — the sponsor's controlled 10-term taxonomy — was the one
variant that improved monotonically across the official tune set, a 1000-POI
stress corpus, and a 150-query synthetic eval set (see "Stress-testing corpus"
below). Out-of-vocabulary query tokens are also canonicalized on the BM25 side,
independent of the parser's own typo corrector: a ≥4-char token absent from the
index's own lexicon snaps to its unique edit-1 neighbour before scoring;
ambiguous tokens and clean, fully in-vocabulary queries are left untouched.

## Typo tolerance

The rule parser's typo corrector runs unconditionally, in every deployment mode
(local and cloud alike — it needs no network). It matches out-of-vocabulary query
tokens against the closed category/attribute/gazetteer vocabulary using
**optimal-string-alignment (Damerau) edit distance ≤ 1**, not plain Levenshtein —
a single adjacent-character transposition (e.g. "tihn" → "tinh") counts as one
edit, which plain Levenshtein distance (2, for that pair) cannot correct. A
two-tier tie-break prefers a unique transposition match over a substitution match
when both exist; same-type collisions still refuse rather than guess. Corrections
only ever land on the ~127-token closed vocabulary (category keywords, attribute
taxonomy, gazetteer names); a much broader ~782-token "known" set (POI-name
tokens, stopwords, vendored Vietnamese common words) defines what counts as
out-of-vocabulary in the first place, so a real subject word is never hijacked
into an unrelated vocabulary term. Entirely local and deterministic — no LLM
involved.

**Hard constraints (post-rank, relaxing).** When a query *explicitly* names a
constraint, results must honor it: a pure category ("cà phê") returns only that
category, a pure location ("quận 1") only that district, a distinctive subject
("bún chả") only POIs named for it. These filters are applied **after** ranking,
over the already-sorted list, and **relax most-specific-first** until ≥1 result
survives — so they enforce "only" without deleting recall or ever returning empty.
A coverage gate (fire the category filter only when the parse fully explains the
query) keeps mis-parses like "nơi mua sắm có nhiều nhà hàng…" from over-filtering.

## The 9 signals — 6 map 1:1 to the sponsor's `Ranking_Signals`, + `category` + `price`

| Sponsor signal | Our signal | Definition |
|---|---|---|
| `relevance_score` | **semantic** | hybrid BM25+dense RRF relevance, calibrated to [0,1] by a fixed max (not per-query min-max) |
| `distance_score` | **distance** | `exp(-d_km/3)` from the resolved anchor; diagnostic value 0.5 but weight excluded without one |
| `rating_score` | **rating** | Bayesian `(v/(v+m))·R + (m/(v+m))·C`, low prior `m` so the narrow 3.8–4.7 band still varies |
| `popularity_score` | **popularity** | `popularity/100` |
| `review_signal` | **review** | query need-terms matched against POI tags + description (distinct from structured attributes) |
| `business_attributes` | **attributes** + **open_now** | taxonomy attribute match, plus the time dimension: open at the (injected) query time, handling `24/7` and overnight hours |
| `freshness_score` | — (documented) | **not implementable** on this dataset (no recency field); a production roadmap item |
| — (our addition) | **category** | 1.0 if the POI matches the parsed category intent, 0.0 on mismatch, 0.5 when no category is parsed — a category-consistency prior so malls/gas stations don't outrank cafés on a "cà phê" query |
| — (our addition) | **price** | affordability preference from `price_level` (1–4): a cheap intent (`rẻ`/`bình dân`) scores cheaper POIs high, an upscale intent (`sang`/`cao cấp`) inverts it; **neutral 0.5 when the query names no price**, so price-less queries are unaffected |

Six of the seven sponsor signals are implemented (business_attributes splits into
the structured-attribute match and the open-now time check); `freshness` is
explicitly accounted for rather than faked. We add two signals beyond the
sponsor's list — **category** (category-consistency) and **price** (affordability
preference) — so the ranker weighs **9 signals** in total. `price` carries a fixed
weight (0.20) rather than a tuned one: only 2 of the 60 eval queries express a price
intent, too few to tune on without over-fitting (NFR-6), and it is neutral on every
query that names no price. Every result keeps a per-signal breakdown, which powers
both the UI bars and the explanations.

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
  ascent (coarse grid, minimum-improvement margin, 0.05 floor so all 8 tuned
  signals stay live). The 9th signal, `price`, carries a fixed 0.20 weight — not
  eval-tuned, since only 2/60 queries express price intent. Test is evaluated
  **once per milestone**. A test
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
| G3 full NDCG@5 / Recall@3 (**test**) | **0.963 / 0.983** (≥0.80 / ≥0.75) |
| G4 warm p95 latency | 9.2 ms (<200 ms) |
| G5 robustness | 138/138 checks pass |

**Ablation (tune):** random 0.005 → BM25 0.861 → dense 0.881 → hybrid 0.922 →
full re-rank **0.959** (NDCG@5) — each stage adds value.

## Honest caveats

- BM25 is unusually strong here because the **synthetic** POI descriptions embed
  the intent words literally; the hybrid/re-rank lift is therefore modest on the
  public set and concentrated in **Discovery** and **no-category-word intent**
  queries, where the need isn't stated in the text.
- The 20-query test split is small; we report bootstrap CIs (e.g. NDCG@5 0.963,
  95% CI [0.907–1.000]) and per-cell n rather than a bare headline. Small per-cell
  categories (n≤2) are anecdotal and flagged as such in `reports/metrics.md`.

## Stress-testing corpus

Two additional artifacts pressure-test the engine beyond the official 111-POI /
60-query set: a **1000-POI synthetic stress corpus**
(`data/synth/synth_dataset.xlsx` = the official 111 rows verbatim + 889 seeded
distractors following the official data's own distributions — geography,
category mix, attributes, hours) and a **150-query synthetic eval set**
(`data/synth/eval_synth.json`, ground truth by construction, not fitted to the
official queries). `scripts/tune.py --pool extended` optionally folds both into
weight tuning (mean NDCG@5 over the combined 190 tune+synth pairs) alongside the
default `--pool official`; either way the official 20-query test split is never
read by tuning and remains the sole G3 source (NFR-6). Headline number at this
larger scale, pre-Tier-1: rules-arm typo-correction NDCG@5 0.510. The Tier-1
typo/BM25/LLM-gate changes documented above land in this same session; re-run
the stress-corpus eval and see `reports/` for the current post-change figure —
not restated here to avoid hand-writing a number a script should generate.

## Models & provider posture

Local **`BAAI/bge-m3`** (multilingual embeddings) is the primary provider: the build,
tuning, and gates all run against it, so the demo has no hard network dependency (NFR-3).

**Deployment modes** (`src/semsearch/config.py`, env `SEMSEARCH_MODE`) make this posture
switchable per host: `local` (bge-m3 only), `local-first` (local, degrading to the Bedrock
chain if bge-m3 is broken on the host), and `cloud` (Bedrock-only for remote hosting without
the 2.3 GB model; all providers failing degrades to a BM25-only floor, and the LLM parse turns
on by default). `DEFAULT_MODE` is now `cloud` for deployment, but every reported metric still
runs pinned to `mode='local'` + the local provider: every measurement entry point
(`engines.py` factories, `tune.py`, `bench_latency.py`, `sample_queries.py`, `robustness.py`)
pins BOTH `provider='local'` and `mode='local'` explicitly, so neither the embedding space nor
the LLM-parse default can drift with the deployment mode — the integrity guarantee is
unchanged, and a regression test
(`tests/test_integrity.py::test_eval_engines_immune_to_deployment_mode`) enforces it.

**LLM query improvement (FR-4).** Riding the SAME intent-parse call (no extra request), the
LLM also returns a `corrected_query` — the user's text with typos fixed and Vietnamese
diacritics/tone marks restored. It passes a no-op / length / token-overlap guard (a correction
equal to the original, or one sharing no folded token with it, is dropped as a
refusal/hallucination), then REPLACES the raw query for the rule parse, BM25, dense retrieval,
and subject corroboration; the API `query` echo stays the original text and the correction is
surfaced additively as `meta.correctedQuery`. Because it rides the LLM parse, it is off in
every measured path (all of which pin `mode='local'`, LLM parse off), so it too cannot reach a
reported metric. Switch: `SEMSEARCH_QUERY_REWRITE` / `DEFAULT_QUERY_REWRITE` (on by default).

**LLM invocation gate (`SEMSEARCH_LLM_GATE`).** Even when the LLM parse is on, the default
`auto` gate skips the ~1.7s call for a clean, fully in-vocabulary query — no Vietnamese
diacritic missing, no out-of-vocabulary token — since that case measurably gains nothing
(rules 0.959 vs LLM 0.950 NDCG@5 on the official tune set) while the correction's real value
(up to +0.22 NDCG@5 at 1000 POIs) concentrates on degraded queries: stripped diacritics,
typos, mixed language. `always` forces the call on every query, useful when demoing the
correction on an already-clean query. Known gap: a query that carries Vietnamese diacritics
AND mixes in common English words already present in the BM25 lexicon (e.g. from POI tags)
is gated off even though it is genuinely mixed-language.

**Amazon Bedrock is implemented as a selectable provider** — never the default, never
required to run:

- **Embeddings:** `bedrock-cohere` (`cohere.embed-multilingual-v3`) and `bedrock-titan`
  (`amazon.titan-embed-text-v2:0`), both 1024-dim, L2-normalized in-code. A construction-time
  preflight (2 s connect / 10 s read timeout, one attempt) degrades the whole pipeline to
  `local` on any failure, so vector spaces are chosen coherently and never mixed
  (provider-stamped caches enforce this). A per-query failure after construction degrades
  that query to BM25-only ordering (zero dense vector) instead of crashing or mixing spaces.
- **LLM query parse (FR-4):** Claude (Haiku 4.5) via Bedrock `converse`, env-gated with
  `SEMSEARCH_LLM_PARSE=bedrock` (off by default — the contract endpoint stays deterministic,
  NFR-5). Output is validated against the closed category/attribute vocabularies and
  union-merged with the rule parse; rules win conflicts; any failure or a ~3 s timeout falls
  back to the rule parse alone. **Langfuse tracing** wraps every LLM call when
  `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set (silent no-op otherwise).
- **Preflight:** `uv run python scripts/check_bedrock.py` verifies credentials, region, and
  per-model access in seconds.

Measured provider comparison lives in `reports/embedding-choice.md`; rows for providers
unavailable at report time are recorded as such rather than faked. **Measured live on the
event AWS account** (dense-only, tune split): `bedrock-cohere` Recall@5 0.958 / NDCG@5 0.860
vs local `bge-m3` 0.929 / 0.881 — local stays the chosen primary (best NDCG + local-first,
NFR-3); Titan v2 is not offered in ap-southeast-1, so its default region chain starts at
ap-northeast-1 (a per-model chain — cohere keeps venue-proximal Singapore).
