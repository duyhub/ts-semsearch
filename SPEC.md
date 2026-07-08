# Plan A — Implementation Spec: Tasco Semantic Search & Ranking

Code-level blueprint: the "how". **`PRD.md` is the canonical requirements doc ("what & why");
where the two disagree, the PRD wins.** The Claude Code runbook (`RUNBOOK.md`) executes this
spec phase by phase.

## 0. Stack decisions (final)

| Layer | Choice | Fallback |
|---|---|---|
| Language | Python 3.11, uv/venv | — |
| Retrieval | `rank_bm25` (BM25Okapi) + in-memory dense matrix (numpy) | — |
| Embeddings | Amazon Bedrock `cohere.embed-multilingual-v3` (or Titan Embed v2 — pick by measured NDCG) | local `BAAI/bge-m3` via sentence-transformers (offline safety) |
| Query parse LLM | Claude on Bedrock, tool-forced JSON | rule-based parser (always available) |
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
│   └── derived/               # pois.parquet, eval.parquet, embeddings.npy (gitignored)
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
│   ├── search.py              # SearchEngine facade: query → ranked results
│   ├── eval.py                # metrics + ablation runner
│   └── api.py                 # FastAPI app
├── ui/                        # Next.js app
├── tests/                     # pytest; eval-harness tests are the real gate
├── scripts/
│   ├── ingest.py              # build derived data + embeddings
│   ├── run_eval.py            # prints metrics table + writes reports/metrics.json
│   ├── ablation.py            # bm25 / dense / hybrid / +rerank table
│   ├── bench_latency.py       # p50/p95 over eval queries
│   └── make_samples.py        # generates the ≥10 sample-query submission doc
└── reports/                   # metrics.json, ablation.md, samples.md (committed)
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
- Typo canonicalizer (PRD FR-3): after folding + abbreviation expansion, query tokens
  (len ≥ 4) that match no vocabulary entry are fuzzy-matched with edit distance ≤ 1 against
  the closed vocabularies (category keywords, attribute taxonomy, gazetteer names) and
  replaced by the canonical form ("yen tihn" → "yen tinh"). Query side only — documents
  are clean.

## 4. Embedding document composition

`doc_text = f"{name}. {brand}. {category} / {sub_category}. {district}, {city}. " +
"Đặc điểm: " + ", ".join(attributes) + ". " + ", ".join(tags) + ". " + description`

Query side: embed the **normalized query + expanded intent terms** (e.g. append resolved
attribute names) — measured on tune split; keep whichever wins.

Embeddings precomputed at ingest into `embeddings.npy` (111×d); cosine sim at runtime is a
single matvec. Disk-cache query embeddings keyed by text hash.

## 5. Retrieval (`retrieve.py`)

- `BM25Index.search(text, k=30) -> list[(poi_id, score)]` over folded tokens.
- `DenseIndex.search(text, k=30)` cosine over the matrix.
- `rrf_fuse(runs, k=60, c=60)`: standard reciprocal-rank fusion → top-30 candidates.
- Hard filters applied *after* fusion but *before* ranking: category (if confidently parsed),
  city/district, `required_attrs ⊆ poi.attributes` — with a relaxation rule: if hard filter
  yields <3 results, demote newest constraint to soft.

## 6. Ranking (`rank.py`)

Seven signals, mapping 1:1 to the sponsor's `Ranking_Signals` sheet (PRD FR-7). All
normalized to [0,1]:

| Signal | Sponsor signal | Definition |
|---|---|---|
| `semantic` | relevance_score | min-max-scaled cosine sim within candidate set |
| `attributes` | business_attributes | matched required+soft attrs / requested (taxonomy canonical, structured `attributes` field only) |
| `distance` | distance_score | `exp(-d_km / 3.0)` from anchor; 0.5 neutral if no anchor |
| `rating` | rating_score | Bayesian: `(v/(v+m))·R + (m/(v+m))·C`, m=200, C=global mean, scaled from [3.5,5] |
| `popularity` | popularity_score | popularity_score / 100 |
| `open_now` | business_attributes (time) | 1 if open at query time / satisfies `open_after`, else 0.3 (0.5 if unknown) |
| `review` | review_signal | fraction of requested needs (required+soft, folded) found in POI `tags` + `description` — distinct from the structured attributes field |

`freshness_score` is the sponsor's 7th listed signal but the dataset has no recency field —
it is **not implemented**; the methodology write-up documents it as a production roadmap item
(rank recently-verified data higher). Every sponsor signal is thus implemented or explicitly
accounted for.

`LinearRanker(weights).rank(intent, candidates)` → sorted `RankedResult` with per-signal
breakdown retained. Initial weights: semantic .32, attributes .22, distance .15, rating .10,
popularity .08, open .05, review .08.

**Tuning (`tune.py`):** split eval 40 tune / 20 test **stratified by difficulty** (fixed seed,
split committed to repo). Coordinate ascent on NDCG@5 over weight grid (0–0.5 step 0.05,
renormalized). Never evaluate test split during tuning; `run_eval.py --split test` is the
reported number.

## 7. Query parsing (`parse.py`)

1. **Rule parser (always runs):** folded-text regex + gazetteer. Category keywords, district/city
   patterns, attribute canonicalizer hits, "gần X" → anchor lookup (gazetteer = all POI names +
   districts + landmarks from dataset + ~20 hand-added city landmarks like "hồ gươm").
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
2. **LLM parser (Bedrock Claude, structured output):** same QueryIntent JSON; prompt includes
   taxonomy vocab + category list so outputs are closed-vocabulary. 800ms budget → on timeout or
   schema failure, keep rule result. Merge policy: LLM fills fields the rules left null; rules win
   on gazetteer-verified anchors.
3. Cache parses by normalized query (dict + sqlite).

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

**Latency budget:** parse(rules) 5ms + cached-LLM 0ms + BM25 2ms + dense matvec 1ms + rank 2ms
→ **p95 < 150ms** without cold LLM; first-seen query with LLM parse < 1s. `bench_latency.py`
proves it.

## 10. UI (Next.js)

Single page: search box (debounced) → left: ranked cards (name, badges for matched attrs,
score bar chart per signal, reason line) → right: Leaflet map with numbered pins + anchor marker.
Toggle: "Keyword (BM25 only)" vs "Semantic (full)" side-by-side columns — this is the demo money
shot. Secondary route `/metrics`: renders `reports/metrics.json` + ablation table as slides-ready
visuals. Vietnamese UI labels.

## 11. Testing & gates

- `tests/test_normalize.py` — folding, abbreviations ("cf q1 co wifi" → expected tokens),
  typo fuzzy-matching ("cafe yen tihn" → canonical tokens).
- `tests/test_eval.py` — metric math on toy fixtures (hand-computed NDCG).
- `tests/test_parse.py` — ≥15 canonical queries → expected QueryIntent (golden JSON),
  including coordinate-in-query, ambiguous-anchor, brand-query, and old-vs-new-admin-name
  pair cases ("q1 tphcm" ≡ "phường sài gòn" → same anchor/district).
- `tests/test_api_contract.py` — response shape strictly matches PDF PlaceResult; error
  responses match the PDF ErrorResponse shape and code table.
- **Quality gates (enforced in runbook):**
  - G1 BM25 baseline: Recall@5 ≥ 0.55 (tune split)
  - G2 hybrid > max(bm25, dense) on NDCG@5 (tune)
  - G3 full ranker: NDCG@5 ≥ 0.80 and Recall@3 ≥ 0.75 (test split)
  - G4 p95 latency < 200ms warm
  - G5 all 60 queries return ≥1 result and no exceptions (robustness sweep)

## 12. Submission artifacts (generated, not hand-written)

`make_samples.py` → `reports/samples.md`: 12 diverse queries (cover every `query_category` +
difficulty) with top-5 results, scores, reasons. `ablation.py` → `reports/ablation.md`.
Deck pulls straight from these.
