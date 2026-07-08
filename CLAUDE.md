# Tasco Semantic Search & Ranking — AABW 2026 hackathon entry

## What this project is

Competition entry for **Agentic AI Build Week 2026** (Ho Chi Minh City, GenAI Fund), **Mobility
track, problem P7 "AI Semantic Search & Ranking"**, sponsored by **Tasco / Tasco Maps**
(Vietnam's digital map platform). The build happens in a **24-hour window: Jul 11 09:00 → Jul 12
09:00 submission**, Demo Day Jul 12 10:00–16:00 at Galaxy Innovation Park.

**The problem:** Vietnamese users search maps by *need*, not by place name — "quán cà phê yên
tĩnh để làm việc" (quiet café to work from), "nơi hẹn hò ở quận 1" (date spot in District 1).
Keyword search misses this intent. We build an AI retrieval + ranking engine that understands
query meaning and returns relevant POIs **with explanations**.

**What we're building (requirements in PRD.md, design in SPEC.md):** Vietnamese query
understanding (diacritic folding, abbreviation expansion, LLM intent parse with rule-based
fallback) → hybrid retrieval (BM25 + multilingual embeddings, RRF fusion) → interpretable
7-signal linear re-ranker (semantic, attributes, distance, rating, popularity, open-now,
review/tags — mapping 1:1 to the sponsor's published signals) → signal-derived explanations →
FastAPI implementing Tasco's official `/v1/search` contract → Leaflet demo UI with
keyword-vs-semantic side-by-side.

**Why we win:** we report real IR metrics (Recall@K, NDCG@5, MRR) on the labeled eval set with
a proper tune/test split + ablation table — almost no hackathon team measures; and we match the
sponsor's published API contract exactly ("integration-ready" is an explicit judging ask).

## Where to find things

| Resource | What it contains |
|---|---|
| `PRD.md` | **Canonical requirements (what & why).** Goals/non-goals, personas, FR/NFR with priorities, success metrics, deliverables, risks, traceability. Wins over SPEC.md on conflict |
| `SPEC.md` | **Implementation design (how).** Repo layout, data contracts, module-by-module design, signal formulas, tuning protocol, API contract, quality gates G1–G5 (§11) |
| `RUNBOOK.md` | Phase-by-phase execution plan with prompts, gates, parallelization (worktrees for UI/API), contingency table |
| `PLAN.md` | Strategy: why this problem, judge psychology, demo script, pre-event schedule, submission checklist |
| `docs/problem-statement.md` | The official problem statement (objective, deliverables, submission requirements, suggested architecture) |
| `docs/tasco_api.pdf` | **Tasco's API contract we must match**: `/v1/search` params, `PlaceResult` DTO (stable ids, WGS84, diacritics preserved), auth headers, error codes, mock-server notes, submission expectations (OpenAPI spec, client adapter, latency/fallback notes) |
| `data/raw/ai_maps_track2_dataset_participants.xlsx` | Official dataset, 5 sheets: `README`; `POI_Dataset` (111 Vietnamese POIs: name, brand, category, city/district, lat/lon, rating, review_count, popularity, price, hours, `;`-separated attributes, tags, description); `Attribute_Taxonomy` (10 attrs with semantic meanings — canonical vocab for constraint matching); `Ranking_Signals` (7 signals the sponsor expects); `Public_Evaluation` (**60 labeled queries**: expected top POI ids, category, difficulty, skills_tested — our metric ground truth) |
| `data/derived/` | Generated (gitignored): pois.parquet, eval split, embeddings cache |
| `reports/` | Generated metrics, ablation, sample-query artifacts (committed; script-generated only) |

## Further research pointers

- Event portal (already logged in via Chrome): https://aitalent.genaifund.ai — tracks,
  schedule, prizes. Judging rubric + track prizes announced at Jul 11 kickoff; check then.
- Problem source page: https://aitalent.genaifund.ai/tracks/mobility/maps-semantic-ranking
- Full archive of all 65 AABW problems (context on competing tracks, the Built-with-AWS
  stacking rule): `~/Coding/hackathon-planning/` (see `tracks/README.md`, strategy docs in `plans/`)
- **Built-with-AWS qualifier:** using Bedrock as a core component (embeddings via
  `cohere.embed-multilingual-v3` or Titan v2, query parsing via Claude) makes this entry
  eligible for the AWS bonus track. See `~/Coding/hackathon-planning/tracks/built-with-aws/aws-ai-ml/README.md`.
- Embedding fallback model (offline safety): `BAAI/bge-m3` via sentence-transformers.
- Langfuse tracing on LLM calls — sponsor awards judge-picked teams that use it.

## Hard rules

- Python 3.11, uv. `uv run pytest -q` must stay green. Type hints everywhere, no heavy frameworks.
- TDD for core logic: write/extend tests before implementation.
- **NEVER hardcode any mapping from an eval query to POI ids anywhere in src/.** The eval set is
  for measurement. The test split (SPEC §6: stratified 40/20, committed seed) must never
  influence code or weights — evaluate it once per milestone via `run_eval.py --split test`.
- Vietnamese text: preserve diacritics in all API/UI outputs; folding happens only inside indexes.
- Every phase ends by running its gate command (SPEC §11) and reporting the metric table.
- Bedrock calls must have a working local fallback (bge-m3) and a timeout. The demo can never
  depend on network success.
- Commit after each passed gate: "gate GN passed: <numbers>".
- `reports/` artifacts are generated by scripts, never hand-edited.

## Quality gates (summary — details in SPEC §11)

G1 BM25 Recall@5 ≥ 0.55 (tune) · G2 hybrid beats both single retrievers (tune NDCG@5) ·
G3 NDCG@5 ≥ 0.80 and Recall@3 ≥ 0.75 (test) · G4 warm p95 < 200ms · G5 all 60 queries +
adversarial inputs return 200 with ≥1 result.
