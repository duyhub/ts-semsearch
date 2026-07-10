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
query ─▶ normalize (fold diacritics, expand abbreviations) ─▶ parse intent (rules)
      ─▶ BM25 + dense (bge-m3) ──RRF──▶ hybrid relevance
      ─▶ re-rank all POIs by 9 interpretable signals ─▶ faithful explanations
      ─▶ hard-constraint filter (pure category / location / subject) ─▶ never-empty
      ─▶ /v1/search (Tasco contract) + /v1/semantic-search ─▶ demo UI
```

Signals: **6 map 1:1 to the sponsor's `Ranking_Signals`, plus `category`-fit and `price`
preference** we added (9 total). Full write-up + the mapping: [`docs/methodology.md`](docs/methodology.md).
Client adapter (Dart): [`clients/tasco_adapter.dart`](clients/tasco_adapter.dart). OpenAPI: `openapi.json`.

## Results — all five gates green

| Gate | Metric | Result | Threshold |
|---|---|---|---|
| G1 | BM25 Recall@5 (tune) | 0.917 | ≥ 0.55 |
| G2 | hybrid NDCG@5 > max(bm25, dense) (tune) | 0.922 > 0.881 | > |
| G3 | full NDCG@5 / Recall@3 (**held-out test**) | **0.963 / 0.983** | ≥ 0.80 / ≥ 0.75 |
| G4 | warm p95 latency | 1.1 ms | < 200 ms |
| G5 | robustness (60 eval + adversarial) | 138/138 | 0 failures |

Reproduce: `uv run python scripts/report_metrics.py` → [`reports/metrics.md`](reports/metrics.md).
Ablation, embedding choice, latency, sample queries: see [`reports/`](reports/).

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
- Optional: AWS Bedrock credentials — Bedrock is a *selectable, measured* embedding/LLM
  provider (Built-with-AWS eligibility), **not** required to run. Local `bge-m3` + the
  rule parser are the default and the path the demo runs on.

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
#         http://127.0.0.1:8000/admin (read-only pipeline transparency: weights + per-query trace)
#   API:  GET /v1/search?q=...            (Tasco contract)
#         GET /v1/semantic-search?q=...   (+ breakdown, reasons, intent)
#         GET /admin/config               (read-only committed ranking weights)
#         GET /health   ·   GET /docs (OpenAPI)
```

## Verify

```bash
uv run python -m pytest -q               # 94 tests
uv run python scripts/robustness.py      # G5 sweep (60 eval + adversarial)
uv run python scripts/bench_latency.py   # G4 cold vs warm p95
uv run python scripts/sample_queries.py  # -> reports/sample-queries.md
```
