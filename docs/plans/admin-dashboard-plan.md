<!-- /autoplan restore point: ~/.gstack/projects/duyhub-ts-semsearch/dashboard-admin-autoplan-restore-20260710-161732.md -->
# Plan: Pipeline Transparency View (`/admin`) — REFRAMED

Branch: `dashboard-admin` · Owner: Yen · Target: AABW 2026 demo polish

> **Reframe note (autoplan, user-approved at the Phase 1 gate).** The original ask —
> separate port + live-editable weights + streamed SSE logs — was challenged by both
> Claude and Codex (6/6 CEO consensus) and reframed by the user's decision to:
> **(1)** serve on the existing app at `/admin` (no second port), **(2)** show signal
> weights **read-only**, **(3)** deliver a **read-only per-request transparency panel**
> instead of streamed logs. The original scope is preserved in the restore point above.
> Rationale: a mutable weight editor broke NFR-5 determinism (a P0) and the eval-integrity
> hard rule; the explainability data judges score already ships on `/v1/semantic-search`.

## 1. What & why

Build a **read-only pipeline transparency view at `/admin`** on the existing FastAPI
app. It serves two things:

1. **Config view (read-only):** the 9 committed ranking-signal weights, each with its
   normalized share and a one-line description — plus a link to the committed ablation
   sensitivity table. Labeled "tuned on tune split, committed" so the integrity story
   reads at a glance.
2. **Transparency panel (read-only, per-request):** type a query, see the pipeline
   *think* — parsed intent → retrieval/fusion summary → per-signal contribution bars for
   each top result (weight × signal value) → Vietnamese reasons → latency. Every value is
   traceable to the response, nothing mutable.

Judging value: turns the already-shipped `/v1/semantic-search` payload into a visible
"show the machine thinking" moment on a surface judges already look at, with determinism
(NFR-5) and the held-out-eval narrative (NFR-6) intact.

## 2. Current-state facts (verified in code)

- 9 signals in `rank.py`; weights load once at boot via `load_weights()` from committed
  `data/weights.json` → `pipeline.ranker.weights` on `app.state.pipeline`.
- `/v1/semantic-search` (`api.py:240`) already returns per-signal `breakdown`, `weights`,
  `reasons[]`, and the parsed `intent` echo. **The transparency data already flows.**
- `create_app()` already serves `index.html` at `/` and mounts `/ui` static assets — the
  pattern to mirror for `/admin`.
- Determinism (NFR-5): ranker clock fixed to `DEFAULT_EVAL_NOW`. Gate G4: warm p95 < 200ms.
- No logging exists today (greenfield) — kept out of the hot path per the review.

## 3. Constraints this must respect
- **Determinism preserved:** read-only means identical requests stay identical. No mutable
  server state on any path a search reads.
- **Eval integrity:** nothing here writes weights or touches the tune/test split. Weights
  are displayed from `load_weights()`; `run_eval.py` is untouched.
- **Latency (G4):** the panel calls the existing `/v1/semantic-search`; the trace is built
  from already-computed data (`breakdown` exists), so no new hot-path cost. Any optional
  stage-logging emits **one INFO record per request**, level-guarded — never per-POI in the
  inner loop.
- **Contract-exact `/v1/search` untouched:** any new `trace` field lives ONLY on the
  extended `/v1/semantic-search`, and is additive + deterministic.
- Offline-safe, no new dependency (reuse vendored Leaflet/font assets; no SSE).

## 4. Architecture (additive, one app)
- `GET /admin` → serves `ui/admin.html` (mirror the existing `index()` handler; graceful
  fallback text if the file is missing).
- `GET /admin/config` → JSON: `{weights, normalizedShare, signals:[{key, weight, share,
  description}], tunedNdcg5Tune}` sourced from `load_weights()` + `data/weights.json` meta.
  Read-only; no PUT/POST.
- Optional `trace` object added to the `/v1/semantic-search` response (deterministic,
  additive): `{fusion:{bm25Top, denseTop}, subjectFilterFired, anchorGateFired,
  fallbackFired}`. Built in `pipeline.rank_scored`/`search` from data already computed.
  Behind the extended endpoint only.
- `ui/admin.html`: reuses `index.html`'s CSS/components (Be Vietnam Pro already vendored,
  signal-bar fills already styled) so there is one visual language, not two.

### Files
- New: `ui/admin.html`; tests in `tests/`.
- Modified: `src/semsearch/api.py` (`/admin`, `/admin/config`, optional `trace` on
  semantic-search); optionally `src/semsearch/pipeline.py` (assemble the trace object);
  `README.md` (one line: open `/admin`).
- Optional/deferred: `src/semsearch/logging_setup.py` for one-INFO-per-request stage logs,
  level via standard `logging` config / env var (NOT a live SSE stream).

## 5. Tests (TDD)
- `test_admin.py`: `GET /admin` → 200 HTML (and graceful fallback when UI missing);
  `GET /admin/config` → committed weights exactly matching `load_weights()`, shares sum ≈ 1,
  no mutating verbs exposed.
- If `trace` added: `test_pipeline_trace.py` — trace fields present and **deterministic**
  for a fixed query; `/v1/search` (contract-exact) response shape unchanged (no `trace`).
- Existing `uv run pytest -q` stays green; existing contract test unaffected.
- G4 spot-check: warm p95 < 200ms unchanged (no new hot-path work).

## 6. Quality gates
- All existing tests green + new tests pass.
- Determinism: same query → identical `/v1/*` responses (unchanged).
- Manual: start the one uvicorn command, open `/admin`, run a query, read the trace,
  confirm weights render read-only and match `data/weights.json`.

## 7. Explicitly NOT in scope (dropped or deferred)
- **Dropped** (per gate): separate port + dual-server launcher; mutable/live weight editor;
  streamed SSE logs + live log-level control.
- **Deferred → TODOS.md:** auth/RBAC on `/admin`; full structured logging module; editing
  any ranking constants; experiment/metrics tracking.

---

# /autoplan Review

## Phase 1 — CEO Review (Strategy & Scope) · mode: SELECTIVE EXPANSION

### 0A. Premise challenge
| # | Premise (original plan) | Verdict |
|---|---|---|
| PR1 | A dashboard is the right vehicle for transparency in a 24h demo | Accept, reframed to read-only panel |
| PR2 | It must run on a **different port** | **Challenged → user chose same-app `/admin`** |
| PR3 | Make all 9 weights **editable at runtime** | **Challenged (integrity) → user chose read-only** |
| PR4 | Stream logs, configurable by level | **Challenged → user chose read-only transparency panel** |

### 0B. What already exists (leverage map)
| Sub-problem | Existing code to reuse |
|---|---|
| Show current weights | `pipeline.ranker.weights`; `/v1/semantic-search` returns `weights` |
| Per-signal contribution trace | `LinearRanker.signals()` + `breakdown`; `generate_reasons()` |
| Serve `/admin` page | `create_app` already serves `index.html` + mounts `/ui` |
| Consistent styling | `index.html` CSS, vendored Be Vietnam Pro, signal-bar fills |

### 0.5 CEO Dual Voices (both ran, foreground-equivalent, sequential)

**CODEX (strategy challenge):** "The live admin dashboard is not the right bet. The
separate port is self-inflicted complexity, the mutable weight editor is a strategic
integrity risk, and streamed logs are a low-signal distraction. The correct move is a
read-only transparency panel on the existing app." F1 CRITICAL (mutable weights),
F2/F3/F4 HIGH (self-imposed port; wrong transparency problem; opportunity cost),
F5/F6 MED (determinism story; infra looks foolish at hour 23).

**CLAUDE SUBAGENT (strategic independence):** "Reject as scoped." F2 CRITICAL (live editor
breaks NFR-5 P0 determinism + manufactures the eval-integrity risk the hard rule prevents),
F1/F3/F5 HIGH (duplicates shipped explainability; self-imposed port adds CORS + G4 gap;
opportunity cost), F4/F6 MED (hot-path logging cost; unauth mutable editor is anti-
production-ready). Reframe: read-only trace panel, same port, read-only weights.

```
CEO DUAL VOICES — CONSENSUS TABLE
  1. Premises valid?                     NO / NO   → CONFIRMED (flawed)
  2. Right problem to solve?             NO / NO   → CONFIRMED (reframe)
  3. Scope calibration correct?          NO / NO   → CONFIRMED (over-scoped)
  4. Alternatives sufficiently explored? NO / NO   → CONFIRMED (same-port/RO unanalyzed)
  5. Competitive/opportunity risk?       HIGH/HIGH → CONFIRMED (displaces G3/G4/demo)
  6. 6-month trajectory sound?           NO / NO   → CONFIRMED (infra will look foolish)
  6/6 CONFIRMED → USER CHALLENGE → user approved the full reframe.
```

## Phase 2 — Design Review (reframed plan)

Scope is now UI-heavy but small; reviewed at depth (single-reviewer, post-reframe — the
architecture was already vetted twice in Phase 1).

| Dimension | Score | Finding / decision |
|---|---|---|
| Information hierarchy | 8/10 | Lead with the transparency panel (the money moment); config view secondary. |
| Interaction states | 7/10 | Must specify: empty (no query), loading, error (API 400/500), populated. **Auto-fix: enumerate all four in `admin.html`.** |
| Per-signal bars | 9/10 | Show **contribution = weight × value**, sorted desc, so the judge sees *why* the top result won — not raw signal values. |
| Visual consistency | 9/10 | Reuse `index.html` CSS/components + vendored font. Risk: a divergent second design language. **Auto-fix: share styles, don't fork.** |
| Vietnamese/diacritics | 10/10 | Preserve diacritics in all labels (NFR-4); reuse existing VI copy. |
| Integrity signaling | 9/10 | Label weights "tuned on tune split · committed" — turns read-only into a *credibility* feature for judges. |

Structural issues (states, contribution-vs-value) auto-fixed into the plan (P5, P1).
No aesthetic taste decisions rise to the gate.

## Phase 3 — Eng Review (reframed plan)

**Architecture (ASCII):**
```
                       existing FastAPI app (:8000)  [UNCHANGED core]
   ┌──────────────────────────────────────────────────────────────┐
   │  /  index.html   /v1/search (contract-exact, UNTOUCHED)        │
   │  /health         /v1/semantic-search ──(+ optional `trace`)──┐ │
   │                                                              │ │
   │  NEW:  /admin  ──serves──▶ ui/admin.html ──fetch──▶ /admin/config (RO) │
   │                              └────────── fetch ──▶ /v1/semantic-search │
   └──────────────────────────────────────────────────────────────┘
            shares the single in-memory pipeline (app.state.pipeline)
```

| Dimension | Verdict |
|---|---|
| Architecture sound? | YES — purely additive; no new process, no launcher, no shared mutable state. |
| Determinism | PRESERVED — read-only; clock unchanged. |
| Test coverage | `test_admin.py` + (if trace) `test_pipeline_trace.py`; assert `/v1/search` shape unchanged. |
| Performance (G4) | SAFE — trace built from already-computed `breakdown`; O(results). No hot-path regression. |
| Security | `/admin` unauthenticated but **read-only** — exposes only committed weights + runs the same public search. Localhost demo. Note as prod follow-up. |
| Error paths | `/admin` graceful fallback if UI missing; panel renders API error state on 400/500. |
| DRY | Reuses `weights`/`breakdown` already on the response; minimal new backend. |

Failure modes: none critical after reframe. The one CRITICAL from Phase 1 (eval
contamination via mutable weights) is **eliminated** by the read-only decision.

## Phase 3.5 — DX Review (reframed plan)

| Dimension | Verdict |
|---|---|
| Time-to-hello-world | **Improved by reframe:** `uv run uvicorn semsearch.api:create_app --factory --port 8000` → open `/admin`. Zero new commands, no launcher. |
| Endpoint naming | `/admin`, `/admin/config` — guessable, consistent with `/v1/*`, `/health`. |
| Error messages | Reuse existing `ErrorResponse` contract (problem + code + requestId). |
| Docs | Add one README line. |
| Escape hatches | N/A (read-only); optional stage-logging level via standard `logging` config. |

## Decision Audit Trail
| # | Phase | Decision | Classification | Principle | Rationale |
|---|-------|----------|----------------|-----------|-----------|
| 1 | CEO | Structured logging in blast radius, but keep off hot path | Mechanical | P2/P5 | Logging absent; one-INFO-per-request only |
| 2 | CEO | Defer auth/RBAC, full logging module, non-weight constants, experiments | Mechanical | P3 | Outside v1 blast radius |
| 3 | CEO | Port: separate → **same-app `/admin`** | User Challenge | — | Both models; user approved |
| 4 | CEO | Weights: editable → **read-only** | User Challenge (integrity) | — | NFR-5 P0 + hard rule; user approved |
| 5 | CEO | Logging: SSE stream → **read-only transparency panel** | User Challenge | — | Data already on response; user approved |
| 6 | Design | Enumerate all 4 interaction states in `admin.html` | Mechanical | P1 | Completeness |
| 7 | Design | Bars show contribution (weight×value), sorted | Mechanical | P5/P1 | Explains *why*, not raw values |
| 8 | Design | Reuse `index.html` styles, don't fork a 2nd design language | Mechanical | P4 | DRY |
| 9 | Eng | `trace` field additive on `/v1/semantic-search` only; `/v1/search` untouched | Mechanical | P5 | Contract-exact stays exact |
| 10 | Eng | Trace built from computed data, no hot-path cost | Mechanical | P1 | G4 safe |

## Implementation Tasks
- [x] **T1 (P1) — `/admin` route + `ui/admin.html`** — serves page (mirrors `index()`, graceful fallback), 4 interaction states (empty/loading/error/populated), reuses existing CSS tokens + font. Files: `api.py`, `ui/admin.html`. ✅
- [x] **T2 (P1) — `GET /admin/config` (read-only)** — returns committed weights + normalized share + per-signal VI descriptions + tuned NDCG@5; PUT/POST → 405. Files: `api.py`. ✅
- [x] **T3 (P1) — Transparency panel** — search box → `/v1/semantic-search`, renders intent pills, per-signal contribution bars (weight×value/Σw, sorted desc), reasons, latency badge. Files: `ui/admin.html`. ✅
- [x] **T4 (P2) — `trace` on `/v1/semantic-search`** — deterministic summary (bm25Top, denseTop, constraintsEngaged/Applied/Relaxed, anchorGateFired, fallbackFired, resultCount) threaded through `rank_scored` (zero cost when not requested); rendered as a stage strip in the panel; NOT on contract `/v1/search`. Files: `pipeline.py`, `api.py`, `ui/admin.html`. ✅
- [x] **T5 (P1) — Tests** — `test_admin.py`, `test_pipeline_trace.py` (present/deterministic/absent-from-contract), `test_logging.py`. 118 passed. ✅
- [x] **T6 (P2) — One-INFO-per-request logging** — `logging_setup.py`, level via `SEMSEARCH_LOG_LEVEL`, one record per request off the hot path, `propagate=False`. Files: `logging_setup.py`, `api.py`. ✅
- [x] **T7 (P3) — README line** — `/admin` + `/admin/config` documented. ✅

## NOT in scope / Deferred → TODOS.md
Separate port + launcher (dropped); mutable weight editor (dropped); SSE streaming + live
log levels (dropped); `/admin` auth; full logging module; editing ranking constants;
experiment/metrics tracking.
