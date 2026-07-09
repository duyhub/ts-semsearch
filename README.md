# Tasco Semantic Search & Ranking

AI retrieval + ranking engine for Vietnamese POI search. Understands query *intent*
("quán cà phê yên tĩnh để làm việc" — a quiet café to work from), not just keywords,
and returns ranked results **with explanations**. Integration-ready with the Tasco Maps
`/v1/search` contract.

> AABW 2026 hackathon entry — Mobility track P7, sponsored by Tasco Maps.
> Requirements: [`PRD.md`](PRD.md) · Design: [`SPEC.md`](SPEC.md) · Build plan: [`RUNBOOK.md`](RUNBOOK.md)
> Demo UI reference: [`docs/mockup/money-shot.html`](docs/mockup/money-shot.html)
> (static) and [`docs/mockup/money-shot-interactive.html`](docs/mockup/money-shot-interactive.html) (interactive).

_(Full overview, methodology, and API docs land at build Phase 9 — see RUNBOOK.)_

## Requirements

The engine runs **fully local** — the demo has no hard network dependency once set up
(offline resilience is a design guarantee, see PRD NFR-3). The only heavy component is
the local embedding model (`BAAI/bge-m3`); BM25, the 111×1024 dense matrix, the rule
parser, and the API are all lightweight.

### Hardware

| Resource | Minimum | Recommended | Notes |
|---|---|---|---|
| RAM | 8 GB | **16 GB** | `bge-m3` loads to ~2–2.5 GB; 16 GB is comfortable when running the API + Next.js UI + a browser together for the demo |
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

- Python 3.11, managed with [uv](https://github.com/astral-sh/uv)
- Node (for the Next.js demo UI)
- Optional: AWS Bedrock credentials — Bedrock is a *selectable, measured* embedding/LLM
  provider (Built-with-AWS eligibility), **not** required to run. Local `bge-m3` + the
  rule parser are the default and the path the demo runs on.

## Setup

```bash
# 1. install deps (Python 3.11 + uv)
uv sync

# 2. pre-download the local embedding model (one-time, ~2.3 GB)
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# 3. build derived data + embeddings, then run the eval
uv run python scripts/ingest.py
uv run python scripts/run_eval.py --split tune
```

_(Setup commands finalize as the build lands — this reflects the RUNBOOK plan.)_
