# Tasco Semantic Search & Ranking

AI retrieval + ranking engine for Vietnamese POI search. Understands query *intent*
("quán cà phê yên tĩnh để làm việc" — a quiet café to work from), not just keywords,
and returns ranked results **with explanations**. Integration-ready with the Tasco Maps
`/v1/search` contract.

> AABW 2026 hackathon entry — Mobility track P7, sponsored by Tasco Maps.
> Requirements: [`PRD.md`](PRD.md) · Design: [`SPEC.md`](SPEC.md) · Build plan: [`RUNBOOK.md`](RUNBOOK.md)
> Demo UI reference: [`docs/mockup/money-shot.html`](docs/mockup/money-shot.html)
> (static) and [`docs/mockup/money-shot-interactive.html`](docs/mockup/money-shot-interactive.html) (interactive).

## How it works

```
query ─▶ normalize (fold diacritics, expand abbreviations, fix typos ≤1 edit)
      ─▶ parse intent (rules)  [cloud: LLM corrected_query + intent, degraded input only]
      ─▶ BM25 + dense (bge-m3) ──RRF──▶ hybrid relevance
      ─▶ re-rank all POIs by 9 interpretable signals ─▶ faithful explanations
      ─▶ hard-constraint filter (pure category / location / subject) ─▶ never-empty
      ─▶ /v1/search (Tasco contract) + /v1/semantic-search ─▶ demo UI
```

Signals: **6 map 1:1 to the sponsor's `Ranking_Signals`, plus `category`-fit and `price`
preference** we added (9 total). Full write-up + the mapping: [`docs/methodology.md`](docs/methodology.md).
Client adapter (Dart): [`clients/tasco_adapter.dart`](clients/tasco_adapter.dart). OpenAPI: `openapi.json`.

Typo tolerance (Damerau/optimal-string-alignment edit distance ≤1, closed-vocabulary
correction) runs entirely locally and deterministically — no network, no LLM — in every
deployment mode.

## Results — all five gates green

| Gate | Metric | Result | Threshold |
|---|---|---|---|
| G1 | BM25 Recall@5 (tune) | 0.929 | ≥ 0.55 |
| G2 | hybrid NDCG@5 > max(bm25, dense) (tune) | 0.933 > 0.881 | > |
| G3 | full NDCG@5 / Recall@3 (**held-out test**) | **0.962 / 0.983** | ≥ 0.80 / ≥ 0.75 |
| G4 | warm p95 latency | 9.3 ms | < 200 ms |
| G5 | robustness (60 eval + adversarial) | 138/138 | 0 failures |

Tune-split NDCG@5 (full pipeline): **0.971**. Beyond the official gates, the engine is
**stress-tested at 9× corpus density**: a deterministic 1000-POI superset (official 111 +
889 seeded synthetic distractors) plus 150 labeled synthetic queries with ground truth by
construction — warm p95 stays 57 ms at 1000 POIs, and typo-tolerance holds without any LLM
([`reports/stress-1000.md`](reports/stress-1000.md), `uv run python scripts/stress_eval.py`).

Reproduce: `uv run python scripts/report_metrics.py` → [`reports/metrics.md`](reports/metrics.md).
Ablation, embedding choice, latency, sample queries, stress: see [`reports/`](reports/).

## Requirements

The engine runs **fully local** — the demo has no hard network dependency once set up
(offline resilience is a design guarantee, see PRD NFR-3). The only heavy component is
the local embedding model (`BAAI/bge-m3`); BM25, the 111×1024 dense matrix, the rule
parser, and the API are all lightweight.

### Hardware

| Resource | Minimum | Recommended | Notes |
|---|---|---|---|
| RAM | 8 GB | **16 GB** | `bge-m3` loads to ~2–2.5 GB; 16 GB is comfortable running the API + a browser together for the demo |
| CPU | any modern x86/ARM laptop | Apple Silicon (M1+) | **No GPU required** |
| GPU | none | optional | Only speeds up cold-query embedding; not needed at this scale |
| Disk | ~5 GB free | — | PyTorch + sentence-transformers install (~2–3 GB) + `bge-m3` weights (~2.3 GB) |
| Network | one-time | — | Needed only to download the model + dependencies; the demo runs offline afterward |

A MacBook (M-series, 16 GB) — the typical hackathon machine — runs everything with
headroom, fully offline once the model is cached. An 8 GB laptop works if you are not
also running other heavy apps.

### Latency and the cold-query path

The warm p95 target is < 200 ms (PRD NFR-1). "Warm" means the query embedding is cached;
a brand-new query pays one `bge-m3` forward pass:

| Path | CPU (Intel/AMD) | Apple Silicon (MPS) |
|---|---|---|
| Cached / rehearsed query | ~1 ms (matvec only) | ~1 ms |
| Cold, never-seen query | ~100–300 ms | ~20–50 ms |

The server loads the model at startup and pre-warms the cache with the eval + rehearsed
demo queries, so the demo stays snappy and the warm p95 holds. `bench_latency.py` reports
cold and warm p95 separately.

### Software

- Python 3.11, managed with [uv](https://github.com/astral-sh/uv). No Node required —
  the demo UI is a lean single-page app served by the API itself.
- **AWS Bedrock is an implemented, selectable provider** (Built-with-AWS): embeddings via
  `cohere.embed-multilingual-v3` or Titan v2 (`provider='bedrock-cohere'|'bedrock-titan'`),
  and Claude (Haiku 4.5) query parsing behind `SEMSEARCH_LLM_PARSE=bedrock`, with Langfuse
  tracing on LLM calls when `LANGFUSE_*` keys are set. Every Bedrock call has a timeout and
  a coherent local fallback — no credentials means the engine runs exactly as local-only.
  Verify a setup with `uv run python scripts/check_bedrock.py`. Local `bge-m3` + the rule
  parser (`SEMSEARCH_MODE=local`) remain a first-class, fully-offline path; **no AWS
  credentials required**.

## Setup

```bash
# 1. install deps (Python 3.11 + uv)
uv sync

# 2. pre-download the local embedding model (one-time, ~2.3 GB)
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# 3. build derived data + embeddings + the committed eval split
uv run python scripts/ingest.py

# 4. tune weights (tune split only) and reproduce the metrics
uv run python scripts/tune.py
uv run python scripts/report_metrics.py     # -> reports/metrics.md (all gates)
```

## Run the demo (API + UI)

```bash
uv run uvicorn semsearch.api:create_app --factory --port 8000
# then open http://127.0.0.1:8000/   (keyword-vs-semantic money shot)
#   API:  GET /v1/search?q=...            (Tasco contract)
#         GET /v1/semantic-search?q=...   (+ breakdown, reasons, intent)
#         GET /health   ·   GET /docs (OpenAPI)
```

The demo UI requests browser location once without delaying search. When permission is
granted, it sends paired `lat`/`lon` to both search lanes: semantic results are proximity-
ranked, while the BM25 comparison lane remains keyword-ranked and only gains distance/filter
context. If location is denied, unsupported, or times out, the map centers on **Trung tâm Hà Nội**
as a clearly labeled display-only default. Those default coordinates are never sent to the API:
the distance weight is inactive and ranking keeps its normal non-location behavior. An explicit
location in the query still takes precedence. Browser geolocation works on `localhost` or HTTPS origins.

### Deployment modes

The engine has one switch for how it sources models — `DEFAULT_MODE` in
`src/semsearch/config.py`, or env `SEMSEARCH_MODE` (env wins):

| Mode | Embeddings | LLM query parse |
|---|---|---|
| `local` | local bge-m3 only; cloud never contacted | off (deterministic) |
| `local-first` | local, degrading to Bedrock (cohere→titan, region chain) if bge-m3 is broken | off |
| `cloud` (default) | Bedrock only — local never loaded (no 2.3 GB model needed); all-fail → BM25-only floor | **on** by default (Claude, else OpenAI) |

`SEMSEARCH_LLM_PARSE` always wins over the mode default: `off` forces it off, `on`/`bedrock`
force it on (full Bedrock→OpenAI chain), `openai` forces it on pinning OpenAI directly
(skips all Bedrock probes); unknown values warn and stay off.

**LLM invocation gate** (`DEFAULT_LLM_GATE` in `src/semsearch/config.py`, env
`SEMSEARCH_LLM_GATE=auto|always`, `auto` by default): when the LLM parse is on, `auto` skips
the ~1.7s call for a clean, fully in-vocabulary query (no missing diacritic, no
out-of-vocabulary token) — a deterministic, network-free check, since that case gains
nothing measured while the correction's value concentrates on degraded queries. `always`
forces the LLM on every query. Inert whenever the LLM parse itself is off.

**Query rewrite** (`DEFAULT_QUERY_REWRITE` in `src/semsearch/config.py`, env
`SEMSEARCH_QUERY_REWRITE=on|off`, on by default): rides the LLM parse — its `corrected_query`
(typos fixed, diacritics restored) replaces the raw text for the rule parse, BM25, dense
retrieval, and subject corroboration. The `query` echo stays the original; the correction is
surfaced additively as `meta.correctedQuery` (only when it differs). Inert whenever the LLM
parse is off/unavailable, so no measured path is affected.

Remote hosting without the local model: `SEMSEARCH_MODE=cloud` plus AWS credentials
(embeddings + Claude) and/or an OpenAI key (`OPENAI_API_KEY` or the gitignored
`.env/OPENAI-API-key.txt`) for the LLM parse. `GET /health` reports what actually
resolved (`mode`, `embeddings`, `llm_parse`, `llm_gate`, `query_rewrite`);
`uv run python scripts/check_bedrock.py` previews all three modes against live credentials.

## Deploy (Railway)

Public demo link for judges — no GPU, no model download. Connect the repo on
[railway.com](https://railway.com); Railway auto-detects the `Dockerfile` (config in
`railway.json`). The image ships `SEMSEARCH_MODE=cloud` (the code default), so embeddings
come from Bedrock and the local `bge-m3`/torch stack is excluded — the image is ~650 MB and
cold-boots in ~5 s (`/health` gate, `healthcheckTimeout` 120 s covers Bedrock prewarm).

Required service variables (Bedrock embeddings + Claude parse):

- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — plus `AWS_SESSION_TOKEN` only for temporary creds.

Optional:

- `OPENAI_API_KEY` — LLM-parse fallback when Bedrock Claude is unavailable.
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` — LLM tracing.

`GET /health` reports what actually resolved (`mode`, `embeddings`, `llm_parse`). With no AWS
creds the service still boots and serves the deterministic BM25-only floor (degraded, never a
crash-loop). Railway injects `$PORT`; the container binds `0.0.0.0` and honors it.

## Verify

```bash
uv run python -m pytest -q               # full suite
uv run python scripts/robustness.py      # G5 sweep (60 eval + adversarial)
uv run python scripts/bench_latency.py   # G4 cold vs warm p95
uv run python scripts/sample_queries.py  # -> reports/sample-queries.md
```
