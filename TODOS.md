# TODOS

Deferred items surfaced during the engineering plan review (`/plan-eng-review`, 2026-07-08).
Each carries enough context to be picked up cold. Priority: do during the referenced phase if
time allows; none blocks a passing gate.

---

## TODO-1 — Ablate the Bedrock LLM parser's retrieval contribution

- **What:** Add one row to `scripts/ablation.py` → `reports/ablation.md`: NDCG@5 / Recall on the
  tune split with rules-only parse vs rules+LLM parse.
- **Why:** The plan tests the LLM parse for *correctness* (golden tests) but never measures whether
  it improves *retrieval*. With local-first (D1) the measured demo path is no-LLM, so the LLM's
  quality contribution is currently unknown and undefendable — a judge asking "does the LLM help
  ranking?" has no answer.
- **Pros:** Makes the Bedrock/AWS-bonus component evidence-backed; one row; strengthens the ablation
  story.
- **Cons:** Requires a tune-split run with the LLM on (Bedrock calls, minutes, small cost).
- **Context:** FR-4 / FR-10. State plainly that the LLM parser's *primary* justification is the
  Built-with-AWS bonus (goal G-F); the ablation row shows its measured effect on top of that.
- **Depends on:** Phase 5 (parse.py) + Phase 4 (ablation harness).

## TODO-2 — Right-size the Bayesian rating prior `m`

- **What:** Lower `m` from 200 to a fixed low value (~20–50); inspect the smoothed-rating
  distribution across the 111 POIs to confirm the `rating` signal actually varies.
- **Why:** With `m=200`, a POI needs 200 reviews just to reach a 50/50 blend with the global mean;
  on a 111-POI synthetic set nearly every POI is pulled hard to the mean, flattening the signal
  toward useless. A signal that doesn't vary can't help ranking or explanations.
- **Pros:** Restores a working `rating` signal (one of the 7 you pitch); ~5-minute fix once you see
  the distribution.
- **Cons:** `m` is another knob — set a sensible fixed value, do **not** grid-search it (that would
  compound the selection-multiplicity risk from A3).
- **Context:** SPEC §6 rating formula. Print the smoothed-rating spread during Phase 4.
- **Depends on:** Phase 4 (rank.py).

## TODO-3 — Make the offline / 3-command reproducibility claim true

- **What:** Either commit the small provider-stamped doc-embedding matrix (~0.5MB) + split, OR write
  an explicit README "one-time online build step (model fetch + ingest)" and soften NFR-7 wording to
  "≤3 commands after a one-time model fetch."
- **Why:** NFR-7 promises "metrics reproducible from a fresh clone in ≤3 commands" and NFR-3 promises
  offline resilience, but `derived/` is gitignored and bge-m3 (~2GB) isn't committed — a fresh clone
  silently needs a network download. The gap is invisible until a judge (or you, at the venue on
  flaky wifi) clones fresh.
- **Pros:** Makes the reproducibility + offline claims actually true; protects venue setup.
- **Cons:** Committing the matrix adds a small binary (the model itself stays external); documenting
  the build step is free but keeps the download.
- **Context:** SPEC §1 (`derived/` gitignored), NFR-3, NFR-7. The doc matrix is deterministic given
  the committed model + data, so committing it is defensible.
- **Depends on:** Phase 3 (embeddings.py) + the provider-stamp fix (A2).

---

## TODO — Admin transparency view: deferred scope (autoplan, 2026-07-10)

Surfaced while reframing the `/admin` pipeline-transparency view (docs/plans/admin-dashboard-plan.md).
The mutable-weights / separate-port / SSE-logging scope was dropped at the CEO gate (both models
flagged it; user approved a read-only reframe). These remain out of scope for v1:

- **Auth/RBAC on `/admin`** — v1 is localhost + read-only (only exposes committed weights and runs
  the public search), so no auth. Needed before any non-localhost deploy.
- **Full structured-logging module** — v1 keeps at most one INFO stage-summary per request off the
  hot path. A ring-buffer/SSE/live-level observability stack was rejected as low-signal for judging.
- **Editing ranking constants** (COS_LO, RRF_C, radii, weights) at runtime — deliberately excluded;
  any change must go through the offline tune path that writes committed `data/weights.json` from the
  tune split only (protects NFR-5 determinism + the eval-integrity hard rule).
- **Experiment / metrics tracking, historical log storage** — out of scope; the deck + reports/ carry
  metrics.
