# Build Plan A — Tasco Maps: AI Semantic Search & Ranking (Mobility P7)

**Goal:** win the Mobility track (and stack Built-with-AWS) with a hybrid semantic retrieval +
explainable re-ranking engine over Vietnamese POIs, backed by measured IR metrics.

## Why this is winnable

- Dataset is already in hand: 111 POIs, attribute taxonomy (10 attrs), 7 named ranking signals,
  and **60 labeled eval queries** (`Public_Evaluation`: expected top POI IDs, difficulty,
  skills tested). We can report Recall@K / NDCG@5 / MRR — almost no hackathon team shows real
  offline metrics.
- The `Public_` prefix implies a **private eval set at judging** → build to generalize
  (no hardcoding of query→POI answers; everything must go through the pipeline).
- The API PDF spells out exactly what "integration-ready" means. Matching their `/v1/search`
  contract + OpenAPI spec + Dart/REST adapter is cheap for us and scores sponsor points.

## Architecture

```
                       ┌────────────────────────────────────────────┐
 user query (vi)  ──►  │ 1. QUERY UNDERSTANDING                     │
                       │  - normalize: lowercase, tone/diacritic    │
                       │    handling, abbreviation expansion        │
                       │  - LLM parse → structured intent JSON:     │
                       │    {category, location_anchor, required_   │
                       │     attributes[], soft_prefs[], open_after,│
                       │     price_level}                           │
                       │  - rule-based fallback parser (never block)│
                       └───────────────┬────────────────────────────┘
                                       ▼
                       ┌────────────────────────────────────────────┐
                       │ 2. HYBRID CANDIDATE RETRIEVAL (top ~30)    │
                       │  - BM25 over normalized name/brand/addr/   │
                       │    tags/attrs (diacritic-folded index)     │
                       │  - Dense: multilingual embeddings          │
                       │    (Bedrock Titan Embed v2 or Cohere       │
                       │    Embed Multilingual; POI vectors         │
                       │    precomputed, in-memory — 111 docs,      │
                       │    no vector DB needed)                    │
                       │  - Reciprocal Rank Fusion (RRF)            │
                       └───────────────┬────────────────────────────┘
                                       ▼
                       ┌────────────────────────────────────────────┐
                       │ 3. RE-RANKING (interpretable linear score) │
                       │  score = w1·semantic_sim                   │
                       │        + w2·attribute_match (taxonomy;     │
                       │           hard constraints filter first)   │
                       │        + w3·distance_score (geodesic to    │
                       │           resolved anchor via /v1/geocoding│
                       │           or dataset lat/lon)              │
                       │        + w4·rating (Bayesian-smoothed by   │
                       │           review_count)                    │
                       │        + w5·popularity + w6·open_now       │
                       │  weights tuned on Public_Evaluation via    │
                       │  grid search / coordinate ascent           │
                       │  (optional: LLM cross-encoder rerank of    │
                       │   top-10 as an ablation, not the default)  │
                       └───────────────┬────────────────────────────┘
                                       ▼
                       ┌────────────────────────────────────────────┐
                       │ 4. EXPLANATION LAYER                       │
                       │  - reasons derived ONLY from matched       │
                       │    signals (traceable: "wifi ✓ from        │
                       │    attributes; 400m from Hồ Gươm; 4.6★")   │
                       │  - LLM phrases them in natural Vietnamese; │
                       │    it cannot invent facts                  │
                       └───────────────┬────────────────────────────┘
                                       ▼
                       ┌────────────────────────────────────────────┐
                       │ 5. SEARCH API (FastAPI)                    │
                       │  - GET /v1/search  — Tasco-contract-       │
                       │    compatible (q, lat, lon, radiusMeters,  │
                       │    category, limit, lang) → PlaceResult[]  │
                       │    stable ids, WGS84, diacritics preserved │
                       │  - GET /v1/semantic-search — extended:     │
                       │    score breakdown per signal + reasons    │
                       │  - Bearer / X-API-Key configurable         │
                       │  - auto OpenAPI/Swagger                    │
                       └───────────────┬────────────────────────────┘
                                       ▼
              ┌───────────────────┐   ┌──────────────────────────────┐
              │ EVAL HARNESS      │   │ DEMO UI (Next.js + Leaflet)  │
              │ Recall@3/5, NDCG@5│   │ - live search box + map pins │
              │ MRR, per-difficulty│  │ - result cards: score-       │
              │ ablation table    │   │   breakdown bars + reason    │
              │ (BM25 / dense /   │   │   chips                     │
              │  hybrid / +rerank)│   │ - "keyword vs semantic"      │
              └───────────────────┘   │   side-by-side toggle        │
                                      └──────────────────────────────┘
```

**AWS stacking (Built-with-AWS qualifier):** embeddings + query-parse LLM (Claude) via **Amazon
Bedrock** as core components; deploy API on App Runner/Lambda; mention Bedrock Guardrails on the
LLM step. Add **Langfuse** tracing on LLM calls (they award judge-picked teams).

### Key design decisions & risk controls

| Decision | Rationale |
|---|---|
| In-memory vectors (numpy/FAISS), no vector DB | 111 docs; a vector DB is résumé noise + a live-demo failure risk. Say this out loud in the pitch — judges respect right-sizing. Architecture doc shows the OpenSearch/pgvector swap-in path for scale. |
| Rule-based parser fallback behind the LLM parse | Demo never dies if Bedrock hiccups; also gives a latency story (<150 ms p50 without LLM, LLM parse cached). |
| Hard constraints filter, soft prefs score | "quán cà phê **yên tĩnh** để làm việc" must never return a loud bar ranked #1 — hard-filter category, soft-score ambience. |
| Explanations derived from signals, LLM only phrases | Kills the hallucination question in Q&A; every reason is auditable. |
| Tune weights on public eval, report on a held-out split | We split the 60 queries (e.g. 40 tune / 20 test) and say so — protects against the private eval and shows ML maturity. |

### Vietnamese-specific edge (your moat)

- Diacritic folding both ways: index folded + raw; queries often arrive un-accented ("ca phe yen tinh").
- Abbreviation/slang expansion: "HCM/SG → TP.HCM", "cf/cafe → cà phê", "q1 → Quận 1".
- Location anchors in Vietnamese ("gần hồ gươm", "quận 1") → resolve to coordinates → distance signal.
- Old↔new admin names after the July-2025 restructuring ("phường sài gòn" ≡ "quận 1, tp.hcm") —
  a curated alias table resolves both eras to the same place; strong live-demo moment since
  mainstream map apps still fumble the new ward names.

## Build plan

### Pre-event (Jul 5–10) — build the engine, ~2–3 h/day

| Day | Deliverable |
|---|---|
| Jul 5 | Repo scaffold; xlsx loader → clean POI/attribute/eval dataframes; **eval harness first** (Recall@K, NDCG@5, MRR, per-difficulty report). |
| Jul 6 | Baseline 1: BM25 + diacritic folding + abbreviation dict. Baseline 2: dense embeddings (pick model by measured NDCG: Titan v2 vs Cohere multilingual vs bge-m3 local). Record ablation numbers. |
| Jul 7 | RRF hybrid; re-ranker with all 6 signals; weight tuning on tune-split; hard-constraint filtering. Target: NDCG@5 ≥ 0.85 on test split. |
| Jul 8 | LLM query parser on Bedrock (structured output) + rule fallback; explanation layer. Attend workshops; ask Tasco people about private eval + judging. |
| Jul 9 | FastAPI `/v1/search` per contract + OpenAPI + tiny Dart/REST adapter snippet; deploy to App Runner. Bedrock AgentCore workshop (useful even here). |
| Jul 10 | Demo UI (Next.js + Leaflet), side-by-side mode, score-breakdown bars. Deck skeleton. Record backup demo video. |

### Event 24 h (Jul 11 09:00 → Jul 12 09:00)

| Hours | Work |
|---|---|
| 09–12 | Kickoff: capture judging rubric + any new data/rules; re-verify pre-built work is allowed; adapt scope to rubric. |
| 12–18 | Integrate any newly released data/queries; re-tune weights; fix Vietnamese edge cases found by teammates hammering the demo. |
| 18–24 | Polish UI; final ablation table; latency benchmark (p50/p95); Langfuse traces screenshot. |
| 00–04 | Deck: problem → live demo plan → architecture → **metrics table** → integration-readiness → roadmap. 10 sample queries doc (submission requirement). |
| 04–07 | Full dress rehearsal ×2; re-record backup video; freeze code. |
| 07–09 | Submit early (07:30, not 08:59). Sleep 60–90 min if possible. |

### Demo script (3 minutes)

1. Type `quan cafe yen tinh lam viec gan q1` (no diacritics, slang) — show keyword search failing side-by-side, semantic ranking nailing it with reason chips.
2. `nơi hẹn hò lãng mạn có view đẹp` — intent-based discovery, no category word at all.
3. Flip to the **metrics slide in the UI itself**: NDCG@5, Recall@3, ablation bars, p95 latency.
4. Close: "implements Tasco's /v1/search contract — this drops into their Flutter app with a one-line base-URL change."

## Submission checklist (from problem statement + API doc)

- [ ] Deck  - [ ] Live demo + recorded video backup  - [ ] Repo + README (setup, tech)
- [ ] ≥10 sample queries with ranked results (generate from eval harness output)
- [ ] Retrieval & ranking methodology write-up + signal descriptions
- [ ] OpenAPI spec  - [ ] Client adapter example  - [ ] Latency/fallback/provenance notes
