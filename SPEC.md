# Plan A — Implementation Spec: Tasco Semantic Search & Ranking

Code-level blueprint: the "how". **`PRD.md` is the canonical requirements doc ("what & why");
where the two disagree, the PRD wins.** The Claude Code runbook (`RUNBOOK.md`) executes this
spec phase by phase.

## 0. Stack decisions (final)

| Layer | Choice | Fallback |
|---|---|---|
| Language | Python 3.11, uv/venv | — |
| Retrieval | `rank_bm25` (BM25Okapi) + in-memory dense matrix (numpy) | — |
| Embeddings | **local `BAAI/bge-m3`** via sentence-transformers (primary — build/tune/G3 run against this; can't be killed by wifi) | Bedrock `cohere.embed-multilingual-v3` / Titan v2 as an env-selectable, **measured** provider (Built-with-AWS core; comparison recorded in `reports/embedding-choice.md`) |
| Query parse LLM | **rule-based parser** (primary, always available; the measured demo path) | Claude on Bedrock, tool-forced JSON — optional enhancement, ablated for retrieval contribution (see TODOS) |

> **Provider posture (eng-review D1):** local-first. The system that is built, tuned, gated
> (G3), and demoed is the local one, so a venue-wifi outage degrades nothing. Bedrock stays a
> real, selectable core component (AWS-bonus eligible) with its numbers recorded; it is never
> the default the gates run against. Both doc-embedding matrices are pre-built and
> provider-stamped (see §4) before rehearsal.
| API | FastAPI + uvicorn | — |
| UI | Next.js + react-leaflet (OSM tiles) | Streamlit single-file (2h emergency build) |
| Deploy | AWS App Runner (or EC2 + caddy) | localhost + ngrok for demo |
| Tracing | Langfuse (LLM calls only) | skip |

## 1. Repository layout

```
tasco-semsearch/
├── CLAUDE.md                  # agent operating manual (see runbook appendix)
├── SPEC.md                    # this file, adapted
├── pyproject.toml
├── data/
│   ├── raw/ai_maps_track2_dataset_participants.xlsx
│   ├── curated/admin_aliases.json  # hand-curated old↔new admin names (committed)
│   ├── eval_split.json         # stratified 40/20 tune/test split (COMMITTED — NFR-6/7)
│   └── derived/               # pois.parquet, embeddings.*.npy (gitignored cache)
├── src/semsearch/
│   ├── data.py                # xlsx → typed frames
│   ├── normalize.py           # Vietnamese text normalization
│   ├── parse.py               # QueryIntent extraction (rules + LLM)
│   ├── embeddings.py          # embed_texts() with provider switch + disk cache
│   ├── retrieve.py            # BM25Index, DenseIndex, rrf_fuse
│   ├── geo.py                 # anchor gazetteer + haversine
│   ├── rank.py                # signal functions + LinearRanker
│   ├── tune.py                # weight search on tune split
│   ├── explain.py             # signal-derived reasons (+ optional LLM phrasing)
│   ├── pipeline.py            # FullPipeline: query → parse → retrieve → re-rank → results
│   ├── eval.py                # metrics + ablation runner
│   └── api.py                 # FastAPI app
├── ui/                        # Next.js app
├── tests/                     # pytest; eval-harness tests are the real gate
├── scripts/
│   ├── ingest.py              # build derived data + embeddings
│   ├── run_eval.py            # prints metrics table + writes reports/metrics.json
│   ├── ablation.py            # bm25 / dense / hybrid / +rerank table
│   ├── bench_latency.py       # p50/p95 over eval queries
│   └── sample_queries.py      # generates the ≥10 sample-query submission doc
└── reports/                   # metrics.json, ablation.md, sample-queries.md (committed)
```

## 2. Data contracts

```python
@dataclass
class POI:
    poi_id: str; name: str; brand: str | None; category: str; sub_category: str | None
    city: str; district: str; address: str; lat: float; lon: float
    rating: float; review_count: int; popularity: float; price_level: int | None
    opening_hours: str | None            # "07:00-22:30"
    attributes: list[str]                # split on ';', normalized
    tags: list[str]; description: str
    doc_text: str                        # composed embedding text, see §4

@dataclass
class QueryIntent:
    raw: str; normalized: str
    category: str | None                 # mapped to dataset categories
    anchor: Anchor | None                # (name, lat, lon) resolved location reference
    required_attrs: list[str]            # hard constraints (from taxonomy vocab)
    soft_prefs: list[str]                # nice-to-haves
    open_after: str | None               # "22:00" for "mở cửa muộn"
    price_max: int | None
    city: str | None; district: str | None

@dataclass
class RankedResult:
    poi: POI; score: float
    breakdown: dict[str, float]          # per-signal contribution (post-weight)
    reasons: list[str]                   # human-readable, signal-derived
```

Eval rows (`Public_Evaluation`): `query_id, input_query, expected_top_poi_ids (";"-sep),
query_category, difficulty, skills_tested` → relevant set = expected ids, order = graded
relevance (first id gain 3, second 2, rest 1) for NDCG. All metric reports break down
per-difficulty **and per-query_category** (PRD FR-9) — the Mixed Language and Discovery
subsets must be visible, not hidden inside the headline number.

**Phase 0 verified data facts (from the actual xlsx — do not re-derive):**
- Columns map: `poi_name→name`, `latitude→lat`, `longitude→lon`, `popularity_score→popularity`.
  `brand` and `price_level` are always present (no nulls) though the dataclass keeps them optional.
- `poi_id` is **bare** (`C001`, `R002`, `S001`, `G010`…); eval `expected_top_poi_ids` uses bare ids.
  The API prepends `poi:` on output (§9). **Never infer category from the id prefix** — prefixes
  (G=72/111) span multiple categories and are opaque.
- `attributes`, `tags`, `expected_top_poi_ids`, `skills_tested` are all `;`-separated;
  `expected_semantic_requirements` and `ranking_signals_to_use` are comma-separated.
- `opening_hours` has **three forms**: `HH:MM-HH:MM`, literal **`24/7`** (always open), and
  **overnight ranges that cross midnight** (`17:00-01:00`, `18:00-03:00`). `open_now` (§6) must
  treat `24/7` as always-open and, when `end < start`, count open if `now ≥ start OR now ≤ end`.
- Ranges: `rating` 3.8–4.7 (narrow — confirms the low-`m` Bayesian prior, TODOS TODO-2),
  `review_count` 120–15 800, `popularity` 50–98, `price_level` 1–4, 4 cities, 12 categories.
- Eval `query_category` counts: Semantic 18, Attribute 14, Intent 8, Location 6, Discovery 6,
  Mixed 5, Category 2, **POI 1** — the n=1/n=2 cells are anecdotal; report n, don't CI an n=1 cell.

## 3. Vietnamese normalization (`normalize.py`)

- `fold(s)`: NFD → strip combining marks, `đ→d`, lowercase, collapse whitespace/punct.
- Abbreviation dictionary (seed; extend from failures found during eval):
  `hcm|sg|tphcm → tp.hcm`, `hn → hà nội`, `q1..q12 → quận N`, `cf|cofe|cafe → cà phê`,
  `ks → khách sạn`, `nh → nhà hàng`, `gần → near-marker`, plus everything in the
  `Attribute_Taxonomy` examples column.
- Both-ways index: BM25 tokenizes folded text; raw preserved for display (API doc requires
  diacritics preserved in responses).
- Attribute canonicalizer: map free text → taxonomy vocab ("yen tinh" → "yên tĩnh").
- Mixed vi/en queries are first-class, not adversarial (PRD FR-3): 5/60 eval queries are
  `Mixed Language Search`. Normalization passes English terms through untouched (multilingual
  embeddings handle them); the abbreviation dict maps common English category words
  ("coffee shop" → "quán cà phê", "hotel" → "khách sạn") for the BM25 side.
- **Single canonicalization module (eng-review C2).** The attribute canonicalizer, category/typo
  canonicalizer, and gazetteer/abbreviation matcher are the same operation — fold a token, match it
  against a closed vocab, replace with the canonical form. Implement one primitive
  `canonicalize(token, vocab, max_edit) -> str | None`; the attribute/category/gazetteer matchers
  are thin callers passing their own vocab and **per-vocab** edit threshold. Do not build three
  divergent copies.
- Typo canonicalizer (PRD FR-3, implemented in `normalize.canonicalize` + `parse.Parser`):
  after folding + abbreviation expansion, query tokens (len ≥ 4) that match no vocabulary
  entry are fuzzy-matched against the closed vocabularies (category keywords, attribute
  taxonomy, gazetteer names) and replaced by the canonical form ("yen tihn" → "yen tinh").
  Query side only — documents are clean.
  **Edit distance ≤ 1 means optimal-string-alignment (Damerau) distance, not plain
  Levenshtein** — a single adjacent-character transposition counts as one edit alongside
  substitution/insertion/deletion. This is a root-cause fix, not a cosmetic one: the FR-3
  canonical example "yen tihn" → "yen tinh" is exactly a transposition (`t-i-h-n` →
  `t-i-n-h`); plain Levenshtein scores that at distance 2, so it provably cannot correct it
  at `max_edit=1`.
  **Two-tier tie-break:** when a token sits within one edit of several vocab terms, a
  transposition candidate is preferred over a substitution/indel candidate (same letters in
  the wrong order is a higher-confidence typo signal than a changed letter) — a *unique*
  transposition match wins even when substitution neighbours also exist (`tihn` → `tinh`
  beats the substitution neighbour `tien`). Same-type collisions still refuse: two
  transposition candidates, or two substitution candidates, leave the token uncorrected
  (`banh` → {`binh`, `benh`, `hanh`} stays as typed).
  **Two-vocabulary design:** corrections only ever land on the closed TARGET vocabularies —
  category keywords, attribute taxonomy, and gazetteer names/districts/cities (~127 folded
  tokens) — while a much broader KNOWN set decides what even counts as out-of-vocabulary in
  the first place: TARGET plus every POI-name token, the stopword list, and the ~582
  vendored Vietnamese common words (~782 folded tokens total). A token already in KNOWN is
  never a correction candidate, so a distinctive subject word that happens to sit one edit
  from a vocab term (a real POI-name token, a common Vietnamese word) is never hijacked into
  an unrelated meaning — only genuinely out-of-vocabulary tokens are corrected, and only onto
  TARGET. **Guard precision:** a big merged vocab means a valid 4+char word can sit within
  edit-distance 1 of an unrelated canonical term and be silently "corrected" into the wrong
  meaning, changing retrieval before it ever runs. Test: a set of known near-collisions must
  **not** be rewritten.

## 4. Embedding document composition

`doc_text = f"{name}. {brand}. {category} / {sub_category}. {district}, {city}. " +
"Đặc điểm: " + ", ".join(attributes) + ". " + ", ".join(tags) + ". " + description`

Query side: embed the **normalized query + expanded intent terms** (e.g. append resolved
attribute names) — measured on tune split; keep whichever wins.

Embeddings precomputed at ingest into a **provider-stamped** matrix (`embeddings.{provider}.{model_id}.npy`
plus a manifest recording `provider`, `model_id`, `dim`, `n_docs`); cosine sim at runtime is a
single matvec. **The loader asserts the active query provider matches the doc-matrix manifest and
refuses (or rebuilds) on mismatch** — bge-m3, cohere-v3 and titan-v2 are all 1024-d, so a mismatch
would otherwise return silent garbage, not an error (eng-review A2). Disk-cache query embeddings
keyed by `hash(f"{provider}:{model_id}:{text}")`, **never text alone** — a text-only key returns
the wrong model's vector during provider comparison. Test: same text under two providers yields two
distinct cached vectors.

## 5. Retrieval (`retrieve.py`)

- `BM25Index.search(text)` over folded tokens; `DenseIndex.search(text)` cosine over the matrix —
  both score the **full 111-doc corpus** (no top-k cut).
- **BM25 document composition (`retrieve.lexical_doc`), previously unspecified.** A flat,
  folded-token document per POI concatenates `name / brand / category / sub_category /
  district / city / address / attributes / tags / description`, each field contributing its
  text a fixed number of times — BM25F-lite via token repetition (`rank_bm25` reads the
  repeats as raised term frequency; no custom BM25F scorer, no new dependency). **Field
  weights: `attributes` ×2, every other field ×1.** The ablation OVERTURNED the intuitive
  "boost identity fields" hypothesis: boosting `name`/`brand` ×3 REGRESSED official-tune
  NDCG@5 (0.9592 → 0.942) via proper-name token collisions (a POI whose *name* happens to
  match outranks the gold POI on category/attribute intent) — the full pipeline already
  resolves identity via dense retrieval + subject corroboration (below), so lexical headroom
  lives in the INTENT fields, not identity. Boosting `category` carried its own collision
  regression (masked in the aggregate number, visible per-query); a lone `tags` boost hurt
  the synth150 set (§6). `attributes ×2` was the only variant that improved monotonically
  across all three measured corpora: tune@111 NDCG@5 +0.0016 (zero per-query regressions),
  tune@1000 NDCG@5 +0.0213, synth150 NDCG@5 +0.0024 — the principled choice, since
  attributes are the sponsor's controlled 10-term taxonomy rather than free-text
  description/address. `district`/`city` are pinned at weight 1, never 0: dropping them (to
  stop the "trà" ~ "Sơn Trà" fold collision) regressed tune NDCG@5 0.959 → 0.948 — location
  overlap is a genuine lexical signal, and the drink/café fold-collision fix is instead
  carried entirely by the parser's `CATEGORY_KEYWORDS` curation (§7) plus the category
  signal, not by starving the lexical location field.
- **BM25-side OOV canonicalization (`BM25Index._canonicalize_oov`).** Independent of the
  parser's typo corrector (§3, which targets the closed category/attribute/gazetteer
  vocabulary), the BM25 index snaps out-of-vocabulary query tokens (folded, len ≥ 4, absent
  from the index's own term lexicon) onto their unique edit-1 lexicon neighbour before
  scoring — the same `canonicalize()` primitive, applied against the index's own document
  vocabulary instead of the closed vocab. Ambiguous tokens (multiple equidistant
  neighbours) are left untouched; a fully in-vocabulary, clean query is scored
  byte-identically to having no canonicalization step at all. Runs after abbreviation
  expansion and before the district-drop de-pollution.
- `rrf_fuse(runs, c=60)`: standard reciprocal-rank fusion, producing a fused **relevance score for
  every POI** — used to compute the `semantic` ranking signal, **not** as a candidate gate.
- **Re-rank over hybrid, no destructive filtering (OV1 + G3-review).** Hybrid RRF relevance is
  computed for the *entire corpus* (no top-k cut), then the 9-signal ranker RE-ORDERS all POIs using
  that hybrid relevance as its `semantic` signal plus attributes/category/distance/rating/popularity/
  open_now/review/price. Category and required-attributes are **soft signals, not AND-filters** — an
  earlier filter-then-rank design (with <3-survivor relaxation) was measured to *lower* recall
  (ablation: Recall@5 0.954→0.879) by deleting relevant POIs, so it was removed. Because
  `semantic == hybrid relevance`, all-weight-on-semantic reproduces hybrid exactly, so tuning makes
  full ≥ hybrid by construction (ablation confirms: full NDCG@5 0.959 > hybrid 0.922 on tune).
- Test: `full (+re-rank)` beats `hybrid` on the ablation (NDCG@5, Recall@5).
- **Empty-set backstop (eng-review C1, protects G5).** Relaxation loosens filters but cannot
  manufacture a result when retrieval itself is empty (emoji-only, gibberish, or fully
  out-of-vocabulary `q`). If the survivor set is still empty after retrieval + relaxation, return
  top-N by `popularity_score` (or nearest to request `lat`/`lon` when present), and mark
  `meta.source = "fallback"` so it stays honest. A present-but-meaningless `q` is a *valid* request
  and must return ≥1 result (only a missing/empty `q` is a 400). Test: each adversarial input in the
  G5 list returns ≥1 result.

## 6. Ranking (`rank.py`)

Nine signals: six map 1:1 to the sponsor's `Ranking_Signals` sheet, `business_attributes`
also drives `open_now` (its time dimension), plus two of our own additions — `category` and
`price` (PRD FR-7). All normalized to [0,1]:

| Signal | Sponsor signal | Definition |
|---|---|---|
| `semantic` | relevance_score | **fixed, query-independent** transform of the fused RRF/cosine relevance — clamp+rescale a calibrated cosine band (e.g. `[0.2,0.8]→[0,1]`), **NOT per-query min-max**. Min-max within the candidate set forces the top result to 1.0 even on weak matches, inflating confidence, corrupting the explanation bars (a scored dimension), and making tuned weights depend on candidate-set composition (worse private-eval transfer). Test: two candidate sets with the same top POI yield the same semantic score (eng-review OV6) |
| `attributes` | business_attributes | matched required+soft attrs / requested (taxonomy canonical, structured `attributes` field only) |
| `category` | — (our addition) | 1.0 if the POI matches the parsed category, 0.0 on mismatch, 0.5 (neutral) when no category is parsed — a category-consistency prior so malls/gas stations don't outrank cafés on a "cà phê" query. A soft signal, not a hard filter |
| `distance` | distance_score | `exp(-d_km / 3.0)` from anchor; diagnostic value 0.5 but weight excluded if no anchor |
| `rating` | rating_score | Bayesian: `(v/(v+m))·R + (m/(v+m))·C`, C=global mean, scaled from [3.5,5]. **m is a low fixed prior (~20–50, not 200)** — on 111 POIs m=200 shrinks nearly every POI to the global mean and flattens the signal; verify the smoothed-rating spread in Phase 4 (eng-review TODO-2) |
| `popularity` | popularity_score | popularity_score / 100 |
| `open_now` | business_attributes (time) | 1 if open at the **injected reference time** / satisfies `open_after`, else 0.3 (0.5 if unknown). Time is injected as `now: datetime`, never wall-clock: eval passes a committed constant (e.g. Sat 14:00 Asia/Ho_Chi_Minh), the API passes real now — otherwise the same query scored at 10am vs 11pm produces different rankings and G3 stops being reproducible (eng-review A1) |
| `review` | review_signal | fraction of requested needs (required+soft, folded) found in POI `tags` + `description` — distinct from the structured attributes field |
| `price` | — (our addition) | affordability preference from `price_level` (1–4): a cheap intent (`rẻ`/`bình dân`) scores cheaper POIs high, an upscale intent (`sang`/`cao cấp`) inverts it; **neutral 0.5 when the query names no price** (constant across POIs → price-less rankings unchanged) and when `price_level` is unknown. Carries a **fixed 0.20 weight**, never eval-tuned (only 2/60 queries express price — NFR-6) |

`freshness_score` is the sponsor's 7th listed signal but the dataset has no recency field —
it is **not implemented**; the methodology write-up documents it as a production roadmap item
(rank recently-verified data higher). Every sponsor signal is thus implemented or explicitly
accounted for.

`LinearRanker(weights).rank(intent, candidates)` → sorted `RankedResult` with per-signal
breakdown retained. Default weights (pre-tuning, `rank.py:DEFAULT_WEIGHTS`): semantic .30,
attributes .25, category .20, distance .10, rating .10, popularity .05, open_now .10,
review .10, price .20 (`price` is the fixed, un-tuned weight).

**Tuning (`tune.py`):** split eval 40 tune / 20 test **stratified by difficulty** (fixed seed,
split committed to repo). **Regularize selection (eng-review A3):** the test split is only 20
queries — one query ≈ 5 pts of Recall — and three things are selected on the 40 tune queries
(embedding provider, query-side composition, and the 8 tuned weights — every signal except the
fixed-weight `price`), so over-fitting to tune noise is the real risk for private-eval transfer. Use a **coarse** weight grid, **cap** coordinate-ascent
passes, and **prefer round weights** when candidates tie within noise. Never evaluate the test
split during tuning; `run_eval.py --split test` is the reported number. **Report it honestly:**
every gate table shows a **bootstrap CI** on the test-split metric and **per-difficulty /
per-category n** (Hard n≈8), so the headline is "0.80 ±ε, Hard n=8", not a bare point estimate.
G3's 0.80/0.75 is an **internal target, not a sacred gate** — take a quick BM25+dense sanity read
on the tune split before treating it as pass/fail, and never let it block UI start (see RUNBOOK).

**Extended tuning pool (`scripts/tune.py --pool official|extended`, default `official`,
landing alongside the Tier-1 changes above).** `official` is exactly the protocol above —
coordinate ascent scored on the 40-query official tune split only. `extended` additionally
folds in the 150-query synthetic eval set (`data/synth/eval_synth.json`, ground truth by
construction, evaluated against the 1000-POI synthetic stress corpus,
`data/synth/synth_dataset.xlsx` = the official 111 rows verbatim + 889 seeded distractors),
and the objective becomes mean NDCG@5 over the combined 190 tune+synth pairs — a larger,
more diverse sample for the coordinate ascent without touching held-out data. **This does
not relax NFR-6/7:** neither pool option ever reads the official 20-query test split, which
remains the sole source `run_eval.py --split test` (G3) reports from.

## 7. Query parsing (`parse.py`)

1. **Rule parser (always runs):** folded-text regex + gazetteer. Category keywords, district/city
   patterns, attribute canonicalizer hits, "gần X" → anchor lookup. **Gazetteer breadth
   (eng-review OV7):** all POI names + districts from the dataset **plus a broad offline landmark
   extract** (OSM/GeoNames or Wikipedia landmark lists) for the four cities — NOT a ~20-entry hand
   list sized to the public eval. A hand list sized to observed queries both (a) fails to generalize
   to the private set's landmarks (anchor → null → distance signal goes neutral → location queries
   silently degrade with no error) and (b) borders on fitting to the public eval (NFR-6 tension).
   Location-category metrics are reported as their own row so a private-set collapse is visible
   before Demo Day.
   Coordinate detection (PRD FR-2): a decimal lat/lon pair in the query (regex, sanity-bounded
   to Vietnam: lat 8–24, lon 102–110) becomes the anchor directly — nearby-search behavior.
   Ambiguous gazetteer names (PRD FR-2) resolve by fixed policy: (a) city/district context in
   the query, else (b) proximity to request `lat`/`lon`, else (c) highest-popularity candidate.
   Admin-name aliasing (PRD FR-2): the gazetteer loads `data/curated/admin_aliases.json` —
   hand-curated entries `{new_name, old_names[], city, lat, lon, old_districts[]}` covering the
   July-2025 renamed wards/provinces for the four dataset cities (e.g. "phường sài gòn" →
   old_districts ["Quận 1"], centroid coords; "quảng nam" → city "Đà Nẵng"). Matching runs on
   folded text after abbreviation expansion; a new-name hit sets the anchor to the alias coords
   and canonicalizes `district`/`city` to the dataset's (old) naming, so downstream filters
   (§5) and outputs are untouched. Old names are dataset-native and pass through unchanged.
2. **LLM parser (Bedrock Claude, OpenAI fallback, structured output — `llm_parse.py`):** same
   QueryIntent JSON; prompt includes taxonomy vocab + category list so outputs are
   closed-vocabulary. **Timeout: ~2s connect / 3s read, no retries** (`_CLAUDE_TIMEOUT`,
   same discipline on the OpenAI fallback client) — the original "800ms budget" here and in
   PRD FR-4 was aspirational and never matched the shipped client config; corrected to the
   actual value. On timeout, network failure, or schema failure, keep the rule result (the
   demo never stalls on the network — CLAUDE.md hard rule). Merge policy (`merge_intent`):
   LLM fills fields the rules left null (category/attributes/price_pref/open_after only);
   rules win on gazetteer-verified anchors — location (city/district/anchor) is entirely
   rule-owned and never read from the LLM output, since a hallucinated location would
   destructively collapse recall through the hard location filter (§5).
   **Degradation gate (`SEMSEARCH_LLM_GATE=auto|always`, `config.DEFAULT_LLM_GATE="auto"`):**
   in `auto` (default), the LLM is invoked only when the query shows a degradation signal —
   no Vietnamese diacritic anywhere in the text, OR at least one folded token ≥4 chars absent
   from the BM25 lexicon — a deterministic, pure function of the query text alone (no clock,
   no network; NFR-5-safe by construction). A clean, fully in-vocabulary query skips the
   ~1.7s call entirely and is byte-identical to the LLM-off path (no correction, no cache
   entry). Measured on the official tune set: clean queries gain nothing from the LLM (rules
   0.959 vs LLM 0.950 NDCG@5), while the whole measured win — up to +0.22 NDCG@5 at 1000
   POIs — sits on degraded queries (stripped diacritics, typos, mixed language). `always`
   forces the call on every query (useful when demoing the correction on an already-clean
   query). **Known limitation:** a query that carries Vietnamese diacritics AND mixes in
   common English words that already sit in the BM25 lexicon (e.g. an English word present
   in POI tags) is gated OFF even though it is genuinely mixed-language — the gate has no
   signal to catch that case short of an always-on LLM call.
3. Cache parses on disk, keyed by (prompt version, provider, model id, raw query) — a JSON
   file per parse under the embedding-cache root (`LLMCACHE_DIR`), not sqlite; only
   successful (non-None) validated parses are cached, so a transient outage self-heals.

## 8. Explanations (`explain.py`)

For each of top-K: emit reasons only from true facts:
matched attrs ("✓ wifi, ✓ yên tĩnh"), distance ("cách Hồ Gươm 400m"), rating ("4.6★ · 1,560
đánh giá"), open ("mở đến 23:00"). Optional LLM pass rewrites the bullet list into one fluent
Vietnamese sentence — with the hard rule that it may only rephrase provided facts (validated by
checking all numbers/attrs appear in source facts; on violation, fall back to bullets).

## 9. API (`api.py`) — match the Tasco PDF contract

- `GET /v1/search?q&lat&lon&radiusMeters&bbox&category&limit&lang` → `{query, results:
  PlaceResult[], meta}` with PlaceResult exactly as PDF: `id ("poi:C001"), type, name, label,
  address, category, coordinates{lat,lon}, distanceMeters, score, source, tags`. Param
  semantics per PDF: `q` required; `bbox` = `minLon,minLat,maxLon,maxLat`; `limit` default 10,
  max 20; `lang` default `vi`.
- Aliases: `GET /search` and `GET /v1/geocode-search` (both in the PDF).
- Errors use the PDF `ErrorResponse` shape: `{error: {code, message, details}, requestId}` with
  the documented code set — 400 `invalid_request` (e.g. missing `q`), 401 `unauthorized`,
  403 `forbidden`, 404 `not_found`, 408 `timeout`, 429 `rate_limited`, 500 `internal_error`,
  503 `service_unavailable`.
- Headers: echo `X-Request-Id` into `requestId` (generate one if absent); accept `X-Locale`
  and `X-Timezone` (timezone feeds the open_now signal; default `Asia/Ho_Chi_Minh`).
- `GET /v1/semantic-search?...` → extended: adds `breakdown`, `reasons`, `intent` echo.
- `GET /health`. Auth: accept anonymous; honor `Authorization: Bearer` / `X-API-Key` if configured
  via env. Config: `BASE_URL`, `EMBED_PROVIDER`, `BEDROCK_REGION`.
- Auto OpenAPI at `/docs`; export `openapi.json` to repo root (submission artifact).
- Optional P2 (PRD FR-13): `GET /v1/poi/{id}` (alias `/poi/{id}`) with `include=ai_summary`
  served by the explanation layer.

**Latency budget:** the sub-10ms budget (parse-rules 5ms + BM25 2ms + dense matvec 1ms + rank 2ms)
assumes the query is **already embedded**. A genuinely novel query (private eval, a judge's ad-hoc
question) has no cache entry, so dense retrieval runs a **cold bge-m3 forward pass** (~100–300ms on
a laptop CPU), plus a one-time multi-second model *load* if lazy (eng-review P1). Mitigations:
(1) **load the embedding model in the FastAPI startup hook**, never on first request;
(2) **pre-warm the query-embed cache** with all eval + rehearsed demo queries at boot so the demo
stays snappy; (3) `bench_latency.py` reports **cold p95 AND warm p95 separately** — the honest
number for a novel query, not just the warm one. Target: **warm p95 < 200ms** (G4); cold p95
reported as-is. `pipeline.py`/`api.py` take an injected `now: datetime` (A1) — eval passes the committed
constant, API passes real now.

## 10. UI (Next.js) — focused on the money shot (CEO review)

**Effort posture (CEO review 0C-bis):** focused Next.js. Concentrate polish on the ONE screen
judges watch live — the keyword-vs-semantic side-by-side. The full spec below is P0 for that
screen; `/metrics` (FR-15) is **P2 / cut-first** (the deck already carries the numbers, and a route
judges never open is not worth night-of hours).

**Implemented stack (Phase 7 decision):** shipped as a **lean vanilla HTML/CSS/JS single-page app**
(`ui/index.html`) served by the FastAPI app at `/` (single origin, no CORS, no Node build step),
fetching the live `/v1/search?engine=keyword` (keyword lane) and `/v1/semantic-search` (full
pipeline). This delivers the same money-shot screen as the specced Next.js while honoring the
CEO/design review's anti-rabbit-hole intent. Serve with
`uv run uvicorn semsearch.api:create_app --factory`.

Single page. **Layout (design review DD1):** the two result columns own the full screen width —
LEFT "Keyword (BM25)" vs RIGHT "Semantic (AI)", side by side — **this is the demo money shot** and
gets the most polish. The Leaflet map is NOT a third column (three columns are illegible at 1080p
from 5m); it lives in a collapsible panel below the fold that auto-opens for location queries
(anchor detected, e.g. "gần hồ gươm"), where numbered pins + anchor marker actually add meaning.
Search box (debounced) + query chips sit above both columns; the latency badge sits in the results
header.

**Demo money-shot touches (CEO review, SELECTIVE EXPANSION — accepted):**
- **Query chips (Delight-1):** a row of ~8 one-tap canonical queries (covering the scenario
  categories) above the search box. Removes live Vietnamese-typing friction on stage, guarantees
  the rehearsed impressive queries run, and models good intent queries for judges who try their own.
- **Animated re-rank (Delight-2):** on toggle keyword↔semantic, result cards animate to their new
  positions (FLIP/position-keyed transition by `poi_id`). Makes the ranking change *felt*, not
  read — the central pitch in one kinetic moment.
- **Live latency badge (Delight-3):** render the server-reported query time (from response `meta`)
  next to results (e.g. "38ms"). Makes the NFR-1 speed claim + no-vector-DB architecture story
  visible live; pairs with the cold/warm split (P1) so the shown number is honest.
- **Match highlighting (Delight-4):** highlight matched required/soft attributes on each card
  (subtle emphasis, matched terms only — keep it to required/soft attrs so it doesn't add noise)
  so the query→result link is instant. Uses data already in the breakdown. Turns the explainability
  dimension into a visible query→result link at near-zero cost.

### Design spec (design review)

**Result card hierarchy (DD2 — signal breakdown).** Each card, top to bottom: rank number + POI
name (largest text) → matched-attribute badges (✓ wifi, ✓ yên tĩnh) with Delight-4 highlighting →
composite score + the **top-3 signals that drove this result's rank** as labeled colored bars (not
all 9) → one-line Vietnamese reason → rating (`4.6★ · 1.560 đánh giá`) + distance. **Click/hover
expands the full 9-signal breakdown** — the "audit any result" demo beat. Rationale: ~10 visible
cards × 9 bars is illegible clutter at 5m; top-3 + expand serves explainability without
the wall of color (subtraction default).

**9-signal color system.** One fixed **colorblind-safe categorical palette** (e.g. Okabe-Ito or
ColorBrewer Set2), one hue per signal, reused everywhere the signal appears (card bars, expanded
breakdown). Defined as CSS variables (`--signal-semantic`, `--signal-distance`, …). Bars carry a
text label too (color is never the only channel). Same hue = same signal across both columns so the
comparison reads.

**Interaction states (Pass 2 — was unspecified).**

| State | What the user sees |
|---|---|
| Loading | Per-column skeleton cards (not a spinner); latency badge shows "…"; chips stay tappable |
| Empty (`meta.source="fallback"`, C1 backstop) | Honest line: "Không có kết quả khớp — đây là các địa điểm phổ biến gần bạn" + the fallback results; never a bare "No results" |
| Error (API 5xx/timeout) | Inline card in the results area: "Máy chủ đang bận, thử lại" + a retry button; the other column and chips stay usable |
| No anchor (location query, gazetteer miss) | Map uses its labeled display-only default; no coordinates enter ranking and the distance weight is inactive |
| Partial (semantic ready, map tiles slow) | Results render immediately; map panel shows its own loading state independently |

**Typography.** A real Vietnamese-first typeface with full diacritic coverage — **Be Vietnam Pro**
(purpose-built for Vietnamese) for display + body; NOT system-ui / Inter / Roboto as primary. Two
weights max (e.g. 700 display, 400/500 body).

**Legibility / contrast (projector, 5m).** Body text ≥ 18px (this is a projected demo, not a laptop
screen); POI names ≥ 28px; all text ≥ 4.5:1 contrast on its background; dark theme with a single
accent. Diacritics preserved everywhere (never fold in the UI — NFR-4).

**Motion.** Animated re-rank (Delight-2): 250–300ms position transition, ease-out, position-keyed by
`poi_id`; respect `prefers-reduced-motion` (fall back to instant reorder). No decorative motion.

**Responsive scope (Pass 6).** **Projector/desktop-first (≥1280px) is the only supported target for
the demo** — explicitly NOT responsive to mobile (see NOT-in-scope). Stated so no one spends demo
hours on a breakpoint no judge will see.

Secondary route `/metrics` (**P2, cut-first**): renders `reports/metrics.json` + ablation table as
slides-ready visuals. Vietnamese UI labels; diacritics rendered correctly; legible at 1080p from 5m.

## 11. Testing & gates

Core-logic tests are **P0 and written alongside the code (TDD)**, not implied (eng-review T1). The
modules that *are* the product must be explicitly covered:

- `tests/test_normalize.py` — folding, abbreviations ("cf q1 co wifi" → expected tokens),
  typo fuzzy-matching ("cafe yen tihn" → canonical tokens), and **near-collision guard** (known
  4+char words that must NOT be rewritten — C2).
- `tests/test_eval.py` — metric math on toy fixtures (hand-computed NDCG); equal-score tie /
  stable-sort determinism.
- `tests/test_parse.py` — ≥15 canonical queries → expected QueryIntent (golden JSON),
  including coordinate-in-query, all three ambiguous-anchor policy branches (a→b→c), brand-query,
  and old-vs-new-admin-name pair cases ("q1 tphcm" ≡ "phường sài gòn" → same anchor/district).
- `tests/test_retrieve.py` — rrf_fuse ordering; hard-filter correctness; relaxation branch
  (<3 → demote); **a relevant POI at low fusion rank still surfaces** (OV1).
- `tests/test_rank.py` — each of the 9 signals in isolation: Bayesian rating (hand-computed
  fixture), distance decay + neutral-no-anchor, attribute match, **semantic fixed-transform
  invariance** (same top POI across two candidate sets → same score, OV6), and **open_now
  determinism** (identical rankings across two injected `now` times, A1); LinearRanker breakdown
  sums to score.
- `tests/test_geo.py` — haversine against a known-distance fixture; anchor resolution + gazetteer.
- `tests/test_explain.py` — faithfulness validator rejects any reason whose numbers/attrs are not
  in the source facts (FR-8).
- `tests/test_embeddings.py` — provider+model_id cache key: same text under two providers →
  two distinct vectors; loader refuses a doc-matrix/provider mismatch (A2).
- `tests/test_api_contract.py` / `tests/test_robustness.py` — empty-candidate backstop returns ≥1
  result for each G5 adversarial input, flagged `meta.source="fallback"` (C1). (The backstop lives
  in the API layer over `pipeline.py`; the legacy `search.py` facade was removed as dead code.)
- `tests/test_integrity.py` — **NFR-6 guard: tune.py and the committed weights never read the test
  split.** The pitch's entire credibility rests on this; it must be a test, not a promise.
- `tests/test_api_contract.py` — response shape strictly matches PDF PlaceResult; error
  responses match the PDF ErrorResponse shape and code table; bbox parse, limit clamp (>20),
  missing-`q` → 400, and X-Request-Id / X-Locale / X-Timezone header handling.
- **Quality gates (enforced in runbook):**
  - G1 BM25 baseline: Recall@5 ≥ 0.55 (tune split)
  - G2 hybrid > max(bm25, dense) on NDCG@5 (tune)
  - G3 full ranker: NDCG@5 ≥ 0.80 and Recall@3 ≥ 0.75 (test split)
  - G4 p95 latency < 200ms warm
  - G5 all 60 queries return ≥1 result and no exceptions (robustness sweep)

## 12. Submission artifacts (generated, not hand-written)

`sample_queries.py` → `reports/sample-queries.md`: 14 diverse queries (cover every `query_category`
+ difficulty) with top-5 results, scores, reasons. `ablation.py` → `reports/ablation.md`.
Deck pulls straight from these.
