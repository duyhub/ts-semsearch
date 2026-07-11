# PRD — Tasco Semantic Search & Ranking

**Product:** AI-powered semantic retrieval and ranking engine for Vietnamese POI search,
integration-ready with the Tasco Maps platform.

**Context:** Agentic AI Build Week 2026 (GenAI Fund, HCMC) — Mobility track, problem P7,
sponsored by Tasco Maps.

**Status of this doc:** canonical requirements ("what & why"). `SPEC.md` holds the
implementation design ("how"); where the two disagree, this PRD wins (known deltas: §11).

**Primary sources:** `docs/problem-statement.md`, `docs/tasco_api.pdf`,
`data/raw/ai_maps_track2_dataset_participants.xlsx` (all requirements below trace to these).

---

## 1. Overview & Problem

Vietnamese users search maps by **need, not name**: "quán cà phê yên tĩnh để làm việc"
(quiet café to work from), "khách sạn gần biển Đà Nẵng", "cây xăng 24/7". Keyword search
misses this intent — it can't map "yên tĩnh để làm việc" to POIs whose attributes are
`wifi; yên tĩnh; phù hợp làm việc; ổ cắm`, can't resolve "gần hồ gươm" to a distance
constraint, and fails outright on non-accented input ("ca phe yen tinh") or mixed
Vietnamese/English queries.

We build the retrieval + ranking engine that closes this gap: it understands query meaning,
retrieves semantically relevant POIs, ranks them with the sponsor's published signals, and
returns results **with auditable explanations** — behind an API that drops into Tasco's
existing Flutter app with a base-URL change.

**The playing field:** sponsor-provided synthetic dataset of 111 Vietnamese POIs
(12 categories — cafés, restaurants, hotels, malls, ATMs, gas stations, attractions, etc. —
across TP.HCM, Hà Nội, Đà Nẵng, Đà Lạt), a 10-attribute taxonomy, 7 named ranking signals,
and 60 labeled evaluation queries (25 Hard / 30 Medium / 5 Easy across 8 query categories,
including 5 mixed-language). Judging uses a **private eval set** — sponsor-confirmed twice:
"the jury will use different questions or datasets, but they will be similar to the public
evaluation questions" (`docs/mobility-track-briefing-recap.md`) and "during judging, we will
also use Hidden Evaluation Scenarios" (`docs/Q&A.md`) — everything must generalize; nothing
may be fitted to specific queries.

**Data-usage rules (organizer Q&A, `docs/Q&A.md`):** the provided dataset is synthetic and
primary; **public data may be used to enrich or normalize it** — which explicitly legitimizes
our curated additions (admin-name alias table, landmark gazetteer, abbreviation dictionary);
no official Search API exists — teams build their own retrieval/ranking pipeline.

## 2. Goals / Non-Goals

### Goals

- **G-A. Understand intent:** parse Vietnamese need-based queries (with slang, abbreviations,
  missing diacritics, mixed English) into structured intent — category, location anchor,
  required attributes, soft preferences, time/price constraints.
- **G-B. Retrieve & rank on meaning:** hybrid semantic + lexical retrieval, re-ranked by the
  sponsor's 7 published signals, measurably better than keyword baseline.
- **G-C. Explain every result:** each ranked POI carries human-readable Vietnamese reasons
  derived only from verifiable signals — no hallucinated claims.
- **G-D. Be integration-ready:** implement Tasco's `/v1/search` contract exactly (DTOs,
  params, aliases, errors, auth, compatibility rules) with OpenAPI spec and client adapter.
- **G-E. Prove it with numbers:** real IR metrics (Recall@K, NDCG@5, MRR) on a held-out test
  split, per-difficulty breakdown, and an ablation table — the credibility edge no other
  team will have.
- **G-F. Qualify for Built-with-AWS:** Amazon Bedrock (embeddings + query-parse LLM) as core
  components, with local fallbacks so the demo never depends on the network.

### Non-Goals

- **No vector database.** 111 documents fit in a numpy matrix; a vector DB is a live-demo
  risk and résumé noise. The architecture write-up documents the OpenSearch/pgvector
  swap-in path for production scale — that documentation is in scope, the system is not.
- **No autocomplete, routing, or geocoding endpoints.** The Tasco PDF describes the full
  platform surface; P7 is search & ranking. We implement the `/v1/search` family only
  (plus an optional POI-detail endpoint, FR-12).
- **Not replacing the map platform.** We enhance search/discovery on top of Tasco Maps,
  per the problem statement's explicit direction.
- **No user accounts, personalization, or query logging product features.**
- **No training/fine-tuning of models.** Pretrained embeddings + an interpretable linear
  ranker; tuning means weight search, not gradient descent.

## 3. Users & Scenarios

| Persona | Who | What they need |
|---|---|---|
| **U1. Vietnamese map user** (primary) | Person searching Tasco Maps by need, typing fast, often without diacritics, mixing English | Relevant places for vague/intent queries, with reasons they can trust at a glance |
| **U2. Tasco platform engineer** | Maintains the Flutter app and its `PeliasClient`/`SearchSuggestion` service layer | A drop-in `/v1/search` backend: stable IDs, WGS84, exact DTOs, configurable base URL + auth, no UI dependencies |
| **U3. Hackathon judge / Tasco reviewer** | Evaluates on Demo Day; likely runs a private eval set | A live demo that visibly beats keyword search, honest metrics, an auditable methodology, integration-readiness |

### Representative scenarios (all must work end-to-end)

1. **Semantic + attributes:** "quán cà phê yên tĩnh để làm việc" → cafés with
   `wifi`, `yên tĩnh`, `phù hợp làm việc`; reasons name the matched attributes. (eval P001)
2. **Location-aware:** "cafe có wifi gần hồ gươm" → anchor "Hồ Gươm" resolved to
   coordinates; nearby wifi cafés ranked by distance + relevance; reason includes distance. (P002)
3. **Intent, no category word:** "nơi hẹn hò lãng mạn có view đẹp" → romantic restaurants /
   rooftops / cafés via the `lãng mạn` and `check-in` attributes — no category keyword to match.
4. **Non-accented slang:** "quan cafe yen tinh lam viec gan q1" → same results as the
   diacritic form; "q1" expands to "Quận 1".
5. **Utility + time:** "cây xăng 24/7 gần đây" → gas stations with the `24/7` attribute,
   ranked by distance from the provided lat/lon.
6. **Mixed language:** queries blending Vietnamese and English (8% of the eval set) →
   handled by multilingual embeddings + normalization, not a special case that crashes.
7. **Old vs new administrative names:** "cà phê yên tĩnh phường sài gòn" returns the same
   results as "cà phê yên tĩnh q1 tphcm" — the post-2025 ward name and the old district
   name resolve to the same place; result labels keep the dataset's naming.
8. **Integrator flow:** `GET /v1/search?q=...&lat=...&lon=...&limit=10&lang=vi` with a
   Bearer token → contract-exact `PlaceResult[]`; malformed request → contract-exact
   `ErrorResponse` with `requestId`.

## 4. Functional Requirements

Priorities: **P0** = must work for demo + submission; **P1** = the winning edge, cut only
under time pressure; **P2** = stretch.

### Query understanding

- **FR-1 (P0) — Vietnamese normalization.** Diacritic folding for matching (raw text
  preserved for display), abbreviation/slang expansion ("hcm/sg → tp.hcm", "q1 → quận 1",
  "cf/cafe → cà phê", "ks → khách sạn", seeded from the taxonomy's examples column).
  *Acceptance:* unit tests map canonical slang/non-accented inputs to expected tokens.
- **FR-2 (P0) — Structured intent extraction.** From free text extract: category (dataset's
  12-category vocabulary), location anchor (POI/landmark/district/city name → coordinates),
  required attributes (closed vocabulary = the 10-attribute taxonomy), soft preferences,
  open-time constraint, price ceiling. A rule-based parser provides this unconditionally.
  Anchors also resolve from **coordinates typed into the query** ("10.7738, 106.704" →
  nearby-search anchor), and **ambiguous place names** follow a fixed disambiguation
  policy: city/district context in the query → request `lat`/`lon` focus → highest
  popularity (both sponsor briefing expectations).
  **Old and new Vietnamese administrative names are equivalent.** The July 2025
  restructuring abolished the district level and merged/renamed wards ("quận 1, tphcm" ≈
  today's "phường sài gòn, tp.hcm"; Đà Nẵng absorbed Quảng Nam; Đà Lạt became wards of
  Lâm Đồng). Users type either era's names; both resolve to the same anchor and filters
  via a curated alias table. The mapping is many-to-many (one old district → several new
  wards and vice versa), so aliases carry representative coordinates and the parent
  old-district set. Outputs keep the dataset's (old) naming — stable IDs and labels.
  *Acceptance:* golden tests on ≥15 canonical queries, including coordinate-in-query,
  ambiguous-anchor, and old-vs-new-admin-name pairs (same query in both namings → same
  anchor and results); intent fields also spot-checked against the eval sheet's
  `expected_intent`/`expected_semantic_requirements` columns (tune split only).
- **FR-3 (P0) — Mixed-language, typos, and degraded input. IMPLEMENTED.** Non-accented,
  mixed vi/en, all-caps, misspelled, and incomplete queries return sensible results through
  the same pipeline. Mixed Language is a first-class eval category (5/60 queries), not an
  edge case. Typo tolerance has an explicit mechanism — query tokens that match no
  vocabulary entry are fuzzy-matched (optimal-string-alignment/Damerau edit distance ≤1, so
  a single adjacent-character transposition corrects too, not just substitution/indel — see
  SPEC §3) against the closed vocabularies (category keywords, attribute taxonomy,
  gazetteer names). "cafe yen tihn" now genuinely parses to the intended attribute: the
  transposition `tihn`→`tinh` is exactly the case plain Levenshtein-1 provably cannot fix;
  embeddings absorb the rest. **Brand queries** ("highlands gần đây") resolve via lexical
  brand fields + popularity signal — named here so they get tests, not just luck.
  *Acceptance:* metrics reported for the Mixed Language eval subset; unit tests for typo'd,
  brand, and incomplete queries; adversarial sweep (NFR-2).
- **FR-4 (P1) — LLM intent parser.** Claude on Bedrock (OpenAI fallback when no Bedrock
  candidate resolves) with structured output, closed to the taxonomy/category vocabularies,
  layered over FR-2: LLM fills fields rules left empty; rules win on gazetteer-verified
  anchors. Hard timeout **~2s connect / 3s read, no retries** → rule result (the original
  "800 ms" figure here and in SPEC §7 never matched the shipped client config; corrected).
  A deterministic degradation gate (`SEMSEARCH_LLM_GATE=auto|always`, default `auto`) skips
  the call entirely for a clean, in-vocabulary query — the LLM fires only when the query has
  no Vietnamese diacritic or carries an out-of-vocabulary token — so the common case pays no
  extra latency; measured gain concentrates on degraded queries (up to +0.22 NDCG@5 at 1000
  POIs), not clean ones. Parses cached (disk cache keyed by prompt version + provider +
  model + query).
  *Acceptance:* golden tests pass with LLM on and off; kill-switch env var.

### Retrieval & ranking

- **FR-5 (P0) — Hybrid candidate retrieval.** Lexical (BM25 over a flat folded-token
  document spanning name/brand/category/sub_category/district/city/address/attributes/tags/
  description, with `attributes` field-weighted ×2 — see SPEC §5) + dense multilingual
  embeddings over composed POI documents, fused (RRF). **Scores the full corpus, no top-k
  candidate cut** — an earlier "top-30 candidate set" design is superseded: a pre-rank
  top-k cut silently discards relevant POIs that fusion under-ranks, and the 9-signal
  re-ranker needs the full corpus to re-order correctly (ablation-backed, SPEC §5 OV1).
  *Acceptance:* gates G1 & G2 — hybrid beats each single retriever on tune NDCG@5.
- **FR-6 (P0) — Hard-constraint filtering with relaxation.** Confidently-parsed category,
  city/district, and required attributes filter candidates ("quán cà phê **yên tĩnh**" must
  never rank a loud venue #1). If a filter leaves <3 candidates, the newest constraint
  demotes to a soft preference — never return an empty list for a well-formed query.
  *Acceptance:* unit tests for filter + relaxation; G5 sweep confirms no empty results.
- **FR-7 (P0) — Multi-signal interpretable ranking, covering all 7 sponsor signals.** The
  sponsor's `Ranking_Signals` sheet names: relevance, distance, rating, popularity,
  business_attributes, review_signal, freshness. Our linear ranker implements:
  `relevance` (embedding similarity), `attributes` (taxonomy-canonical required+soft match),
  `distance` (decay from resolved anchor or request lat/lon; weight inactive without one),
  `rating` (Bayesian-smoothed by review_count), `popularity` (dataset popularity_score),
  `open_now` (opening-hours vs query time / "mở khuya" constraint — the time dimension of
  business_attributes), and `review_signal` (query needs matched against POI tags +
  description, distinct from the structured attributes field). `freshness` is **documented
  as not implementable** on this dataset (no recency field) and appears in the methodology
  write-up as a production roadmap item — every sponsor signal is either implemented or
  explicitly accounted for. Weights are tuned on the tune split only (FR-9). Every result
  retains a per-signal score breakdown.
  *Acceptance:* gate G3 on the held-out test split; ablation shows re-ranking's contribution;
  methodology doc maps our signals 1:1 to the sponsor's seven.
- **FR-8 (P0) — Explanations.** Each returned POI carries 1–4 Vietnamese reasons derived
  exclusively from true, checkable facts: matched attributes ("✓ wifi, ✓ yên tĩnh"),
  distance ("cách Hồ Gươm 400m"), rating ("4.6★ · 1.560 đánh giá"), hours ("mở đến 23:00").
  *Acceptance:* every reason string traceable to a signal value; unit-tested.
- **FR-9 (P0) — Evaluation harness.** Recall@3/5, NDCG@5, MRR over the 60 labeled queries,
  reported overall and per-difficulty and per-query-category; stratified 40/20 tune/test
  split (fixed seed, committed); ablation runner (BM25 / dense / hybrid / +re-rank);
  all outputs script-generated into `reports/`. **Because the test split is 20 queries (one
  query ≈ 5 pts of Recall), every gate table reports a bootstrap confidence interval and the
  per-cell n (Hard n≈8)** — the headline is "0.80 ±ε, Hard n=8", not a bare point estimate
  (eng-review A3). Weight/provider selection is regularized (coarse grid, capped ascent, round-
  weight tie-break) to limit over-fitting the 40 tune queries and protect private-eval transfer.
  *Acceptance:* metric math unit-tested against hand-computed fixtures; split file committed.
- **FR-10 (P1) — Local embeddings primary, Bedrock as a selectable measured provider**
  (eng-review D1). `BAAI/bge-m3` is the default the build, tuning, and gates (G3) run against, so
  the demo can't be killed by venue wifi. Bedrock (`cohere.embed-multilingual-v3` or Titan v2) is
  an env-selectable provider whose numbers are recorded — the Built-with-AWS core component
  alongside FR-4, without being the default the gates depend on. Both doc-embedding matrices are
  provider-stamped and the cache key includes provider+model_id (no silent cross-provider mixing).
  *Acceptance:* provider comparison recorded in `reports/`; pipeline green with either provider;
  loader refuses a doc-matrix/provider mismatch.

### API

- **FR-11 (P0) — Tasco-contract search endpoint.** `GET /v1/search` with aliases `/search`
  and `/v1/geocode-search`. Full parameter set per the PDF: `q` (required), `lat`, `lon`,
  `radiusMeters`, `bbox` (minLon,minLat,maxLon,maxLat), `category`, `limit` (default 10,
  max 20), `lang` (default vi). Response `{query, results: PlaceResult[], meta}` with
  `PlaceResult` field-exact: `id` ("poi:C001", stable), `type`, `name`, `label`, `address`,
  `category`, `coordinates{lat,lon}` (WGS84), `distanceMeters`, `score`, `source`, `tags`.
  Errors use the contract `ErrorResponse` (`error.code/message/details`, `requestId`) with
  the documented code set (400 invalid_request, 401, 403, 404, 408, 429, 500, 503).
  Auth: anonymous accepted; `Authorization: Bearer` and `X-API-Key` honored when configured.
  Headers: `X-Request-Id` echoed into `requestId`; `X-Locale`/`X-Timezone` accepted
  (timezone informs open-now). `GET /health` for liveness. OpenAPI auto-generated and
  exported to `openapi.json`.
  Note: organizers confirmed the challenge-doc JSON is "a recommended example, not a
  mandatory schema" and the exact format is not evaluated (`docs/Q&A.md` §4). Exact-contract
  match stays P0 **by strategy**, not compliance — it is how we take the "technical design &
  production readiness" judging dimension; extending the schema (as `/v1/semantic-search`
  does) is explicitly allowed.
  *Acceptance:* contract tests assert exact response/error shapes; gate G4 latency.
- **FR-12 (P0) — Extended semantic endpoint.** `GET /v1/semantic-search`: same params,
  response adds per-signal `breakdown`, `reasons[]`, and parsed `intent` echo. This powers
  the demo UI and is the transparency showcase.
  *Acceptance:* covered by the same contract test suite.
- **FR-13 (P2) — POI detail endpoint.** `GET /v1/poi/{id}` (alias `/poi/{id}`) per the PDF,
  with `include=ai_summary` served by the explanation layer.

### Demo UI & submission surface

- **FR-14 (P0) — Demo UI (focused on the money shot; CEO review).** Single page: debounced
  search box; ranked result cards with matched-attribute badges, per-signal score bars, and
  reason line; Leaflet map with numbered pins + anchor marker; **keyword-vs-semantic side-by-side
  toggle** (BM25-only vs full pipeline) — the demo centerpiece, which gets the most polish. Plus
  three accepted demo touches (CEO review, SELECTIVE EXPANSION): **one-tap query chips** for the
  ~8 canonical scenarios (removes live Vietnamese-typing risk on stage), **animated re-rank** on
  toggle (the ranking change is felt, not read), a **live latency badge** from response `meta`
  (makes the speed claim visible), and **matched-term highlighting** on each card (matched
  required/soft attributes emphasized so the query→result link is instant, using data already in
  the breakdown). Vietnamese labels, diacritics rendered correctly, legible at 1080p from 5 meters.
- **FR-15 (P2 — cut-first; CEO review).** Metrics page: `/metrics` route rendering
  `reports/metrics.json` + ablation table as presentation-ready visuals. Downgraded from P1 — the
  deck already carries the metrics, so this route is not worth night-of hours against the demo
  money shot; build only if UI time remains.
- **FR-16 (P1) — Client adapter.** Dart/REST adapter example mapping `PlaceResult` →
  Tasco's `SearchSuggestion` (id→id, label/name→label, category/type→meta,
  address→description, coordinates→coordinates), per the PDF's mapping table.
- **FR-17 (P2) — Observability & deploy.** Langfuse tracing on LLM calls (sponsor awards
  judge-picked teams using it); cloud deploy (App Runner) with localhost + ngrok fallback.

## 5. Non-Functional Requirements

- **NFR-1 (P0) — Latency.** Warm p95 < 200 ms over the 60 eval queries on the demo machine
  (no-LLM path); a first-seen query that DOES invoke the LLM parse pays its ~3 s-timeout budget
  (measured ~1.7 s; result disk-cached, so repeats are instant) — and the degradation gate
  (SEMSEARCH_LLM_GATE=auto) skips the LLM entirely for clean queries, so only degraded inputs
  ever pay it. **The warm number assumes a cached query
  embedding; a novel query pays a cold bge-m3 forward pass (~100–300 ms), so the benchmark reports
  cold p95 and warm p95 separately (eng-review P1), and the embedding model loads at server startup
  with the query cache pre-warmed for eval + rehearsed demo queries.** Numbers included in
  submission notes.
- **NFR-2 (P0) — Robustness.** Every eval query **plus** adversarial inputs (empty string,
  emoji, all-caps no-diacritics, 200-char rambling text, pure-English query, pure address,
  coordinate-only query, unknown city) returns HTTP 200 with ≥1 result (or a contract-valid error for truly invalid
  requests like missing `q`) and zero unhandled exceptions.
- **NFR-3 (P0) — Offline resilience.** The full demo path (query → results → UI) has zero
  hard network dependencies: every Bedrock call has a timeout and a local fallback
  (bge-m3 embeddings, rule parser); map tiles cached by rehearsal. A wifi outage on stage
  degrades quality gracefully, never availability.
- **NFR-4 (P0) — Vietnamese text fidelity.** Diacritics preserved in every API and UI
  output (an explicit Tasco compatibility requirement). Folding/normalization exists only
  inside indexes and matchers.
- **NFR-5 (P0) — Integration compatibility** (PDF "Compatibility requirements"): stable IDs
  across responses; WGS84 coordinates; configurable base URL and auth; no app/UI
  dependencies in the service layer; deterministic behavior on identical input (mock-data
  determinism for tests and demos).
- **NFR-6 (P0) — Evaluation integrity.** No mapping from any eval query to POI ids anywhere
  in source code. The test split never influences code, weights, or vocabularies; it is
  evaluated once per milestone and reported as-is. Protects against the private eval and is
  itself a pitch point.
- **NFR-7 (P0) — Reproducibility.** All `reports/` artifacts generated by committed scripts
  from committed seeds/splits; `uv run pytest -q` green at every commit; metrics
  reproducible from a fresh clone in ≤3 commands.

## 6. Success Metrics

### Quantitative (quality gates — the requirement of record)

| Gate | Metric | Threshold | Split | Verified by |
|---|---|---|---|---|
| G1 | BM25 baseline Recall@5 | ≥ 0.55 | tune | `run_eval.py --engine bm25 --split tune` |
| G2 | Hybrid NDCG@5 | > max(BM25, dense) | tune | `run_eval.py` 3-row comparison |
| G3 | Full-pipeline NDCG@5 / Recall@3 | ≥ 0.80 / ≥ 0.75 | **test** | `run_eval.py --split test` (once per milestone) |
| G4 | Warm p95 latency | < 200 ms | all 60 | `bench_latency.py` |
| G5 | Robustness sweep | 100% HTTP 200, ≥1 result, 0 exceptions | all 60 + adversarial | robustness script |

NDCG grading: first expected id gain 3, second 2, remaining 1. All gate runs also report
per-difficulty (Hard/Medium/Easy) and per-query-category tables — Hard is 42% of the eval
set, so a headline number that hides a Hard-query collapse is not acceptable.

### Qualitative — the official judging dimensions (organizer Q&A, `docs/Q&A.md`)

The organizers confirmed judging focuses on five dimensions and does **not** require IR
metrics — teams "may include their own evaluation metrics, but this is optional." That is
precisely why the gates above stay our requirement of record: self-reported metrics are the
differentiator almost no team will bring, serving the first two dimensions as evidence.

| Judging dimension | How we win it |
|---|---|
| Search relevance & semantic understanding | intent queries with no category word land (scenario 3); hybrid retrieval; measured, not asserted (G1–G3) |
| Retrieval & ranking quality | 7-signal ranker mapped 1:1 to the sponsor's signals; ablation table proving each stage's contribution |
| Explainable ranking | per-signal score breakdown + fact-derived reasons on every result (FR-7, FR-8) |
| User experience | keyword-vs-semantic side-by-side, visibly instant (<200 ms), Vietnamese labels, map UI (FR-14) |
| Technical design & production readiness | Tasco-contract API + OpenAPI + client adapter + fallback/latency/provenance notes (FR-11, FR-16, NFR-3) |

## 7. Deliverables

**Product:** search engine (`src/semsearch/`), API service with `openapi.json`, demo UI,
evaluation harness + `reports/` (metrics, ablation, embedding choice, latency).

**Submission package** (problem statement + PDF expectations):

1. Presentation deck (problem → live demo → architecture → metrics/ablation →
   integration-readiness → roadmap).
2. Live demo + recorded backup video of the exact demo script.
3. Source repo + README (overview, setup, technologies).
4. ≥10 sample queries with ranked results — script-generated, covering every query
   category and difficulty (12 planned).
5. Retrieval & ranking methodology write-up, including the 1:1 mapping of our signals to
   the sponsor's seven ranking signals.
6. Description of AI models used (embeddings, LLM parser) and fallbacks.
7. OpenAPI/Swagger spec + Dart/REST client adapter example.
8. Notes on latency, fallback behavior, and data provenance.
9. Runnable local endpoint (mock/staging equivalent) with deterministic sample data.

## 8. Milestones

| When | Milestone | Exit criterion |
|---|---|---|
| Jul 8 (today) | Scaffold + data layer + eval harness + committed split | pytest green; `run_eval.py --engine random` prints full metric table |
| Jul 9 | Baselines (BM25, dense), hybrid fusion, 7-signal ranker + tuning | G1, G2 pass; G3 evaluated once on test |
| Jul 10 | Intent parsers (rules + Bedrock), explanations, API, UI shell; backup video | golden parse tests green; G4 pass; contract tests green |
| Jul 11 09:00 | Kickoff: capture rubric, verify pre-built work allowed, absorb any new data | scope adjusted same morning |
| Jul 11 day | Integration, robustness (G5), UI polish, re-tune with any new data | all gates G1–G5 green |
| Jul 12 by 07:30 | Submission package complete, submitted early | checklist §7 fully checked; dress rehearsal ×2 done |

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Bedrock quota/latency/outage during demo | Dead demo | NFR-3 by design: env-switch to bge-m3 + rule parser; both paths' metrics recorded in advance (contingency: RUNBOOK) |
| Private eval distributed differently than public | Metrics don't transfer | NFR-6 (no fitting to queries), closed-vocabulary parsing, held-out-split discipline; new queries slot in as a new eval split with zero code changes |
| G3 stuck below 0.80 | Weaker headline | Ship best honest number with per-difficulty breakdown; pivot pitch to measurement rigor + hard-query analysis (judges reward honesty over unverifiable claims) |
| UI time sink (Next.js rabbit hole) | No demo polish | Timebox; Streamlit single-file fallback (~90 min) still shows breakdown bars + map |
| Kickoff rules forbid pre-built code | Lose pre-event work | Repo is spec+data+scaffold; specs re-pasteable as prompts (~30 min loss); verify rules at kickoff Jul 11 |
| On-stage wifi failure | Demo interruption | Everything runs localhost incl. embeddings + cached parses; OSM tiles pre-cached by rehearsal |

## 10. Open Questions

- Judging dimension *weights* and track-prize details — the five dimensions themselves are
  now known (`docs/Q&A.md`; see §6), but their relative weighting isn't; capture at the
  Jul 11 kickoff and re-weight polish effort the same morning.
- Whether organizers release additional/private queries at kickoff — if so they become a
  new eval split (no code changes; re-run tuning).
- Deploy target (App Runner vs localhost+ngrok) — decide Jul 10 after measuring event-wifi
  reliability; the demo must not require the deployed instance (NFR-3).

## 11. Traceability & deltas from SPEC.md

| Requirement | SPEC section | Gate |
|---|---|---|
| FR-1, FR-2, FR-3 | §3 normalize, §7 parse | G1, G5 |
| FR-4 | §7 (LLM parser) | golden tests |
| FR-5 | §4–5 embeddings/retrieve | G1, G2 |
| FR-6 | §5 (filters) | G5 |
| FR-7 | §6 rank/tune | G3 |
| FR-8 | §8 explain | unit tests |
| FR-9 | §2, §6, eval.py | G1–G3 |
| FR-10 | §0, §4 | reports/embedding-choice |
| FR-11, FR-12 | §9 api | G4, contract tests |
| FR-13 | — (PRD-only, P2; PDF §3 POI API) | contract tests if built |
| FR-14, FR-15 | §10 ui | demo rehearsal |
| FR-16, FR-17 | §12 artifacts, §0 stack (tracing/deploy) | submission checklist |
| NFR-1…7 | §9, §11, CLAUDE.md rules | G4, G5 |

**Deltas — requirements this PRD introduced that SPEC.md originally omitted** (all now
applied to SPEC; kept here as the audit trail against the sponsor sources):

1. **`bbox` request parameter** on `/v1/search` (PDF supported-parameter table), plus
   `limit` semantics: default 10, max 20 → SPEC §9.
2. **Contract `ErrorResponse`** shape (`error.code/message/details`, `requestId`) and the
   full error-code table → SPEC §9, §11 tests.
3. **`/v1/geocode-search` alias** and `X-Request-Id`/`X-Locale`/`X-Timezone` header
   handling → SPEC §9.
4. **`review_signal` as a distinct 7th signal** (tags/description match) and an explicit
   `freshness` disposition — FR-7 requires all seven sponsor signals accounted for → SPEC §6.
5. **Mixed-language queries as a first-class requirement** (FR-3), not just an adversarial
   robustness case — the eval set contains 5 Mixed Language queries (8%) → SPEC §3.
6. **Per-query-category metric reporting** alongside per-difficulty (FR-9) — needed to
   catch a Mixed Language or Discovery collapse that headline NDCG would hide → SPEC §2.
7. **Coordinate-in-query anchors, typo fuzzy-matching, ambiguous-anchor disambiguation,
   and explicit brand/incomplete-query handling** (FR-2, FR-3) — from the sponsor kickoff
   briefing (`docs/mobility-track-briefing-recap.md`); "by direction" queries stay out of
   scope (routing, sibling challenges) → SPEC §3, §7.
8. **Old↔new administrative-name aliasing** (FR-2) — the July 2025 restructuring renamed
   Vietnamese wards/districts ("quận 1, tphcm" ≈ "phường sài gòn"); the dataset and eval
   queries may use either era. Team-sourced requirement, not in any sponsor doc → SPEC §7.
