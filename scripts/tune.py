"""Weight tuning via regularized coordinate ascent (SPEC §6; eng-review A3, C4).

Tunes the 8 TUNABLE signal weights on a COARSE grid with a minimum-improvement
margin so we don't chase tune-split noise (A3 regularization). `price` is DELIBERATELY
EXCLUDED (C4): it is a fixed 0.20 affordability-preference weight, never eval-tuned (only
2/60 eval queries express price — too few to inform it, NFR-6). It never enters the working
weight dict here, so it is never scored, never coordinate-ascended, and never written;
load_weights() re-supplies it from DEFAULT_WEIGHTS via its per-key fallback.

Two pools (--pool):

  official  (default)  Tune on the 40-query official TUNE split @ the official 111-POI
                       corpus (never test — NFR-6). Writes data/weights.json (committed)
                       with exactly the 8 TUNABLE keys. This is the reproducible protocol
                       that produces the committed weights — its behavior is frozen.

  extended             Tune on official tune40@111 + the 150 synthetic labelled queries
                       (data/synth/eval_synth.json, ground truth by construction, legal to
                       tune on — NFR-6 protects only the official TEST split). Each query is
                       scored against its HOME corpus (official 40 @ 111, synth 150 @ the
                       1000-POI synth_dataset.xlsx), objective = mean NDCG@5 over all 190
                       (query, home-corpus) pairs. Produces a CANDIDATE only:
                       data/derived/weights_extended_candidate.json (gitignored) + a
                       candidate-vs-committed comparison table on stdout. NEVER touches
                       data/weights.json — the orchestrator decides adoption.

  python scripts/tune.py                 # official (writes data/weights.json)
  python scripts/tune.py --pool extended # extended candidate (writes data/derived only)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.data import EvalQuery, QueryIntent, load_eval, load_pois  # noqa: E402
from semsearch.eval import evaluate, ndcg_at_k  # noqa: E402
from semsearch.pipeline import FullPipeline  # noqa: E402
from semsearch.rank import DEFAULT_WEIGHTS, SIGNALS, WEIGHTS_PATH, load_weights  # noqa: E402
from semsearch.split import SPLIT_PATH, load_split, make_split, select  # noqa: E402

GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # coarse (A3); 0.05 floor keeps every signal live (FR-7)
MARGIN = 1e-3  # require a real improvement before moving a weight (regularization)
MAX_PASSES = 2

# The signals we actually tune: everything except the fixed-weight `price` (C4).
TUNABLE = [s for s in SIGNALS if s != "price"]

# Extended-pool data (the synth arm). eval_synth rows are EvalQuery(**row)-shaped; the
# labels reference SYN#### ids that ONLY resolve against the 1000-POI synth corpus.
SYNTH_CORPUS = Path("data/synth/synth_dataset.xlsx")
SYNTH_EVAL = Path("data/synth/eval_synth.json")
# Candidate output — gitignored data/derived, NEVER data/weights.json (the orchestrator
# owns adoption). Kept a distinct constant so the never-clobber contract is testable.
CANDIDATE_PATH = Path("data/derived/weights_extended_candidate.json")


def tunable_seed() -> dict[str, float]:
    """Pre-tuning weight dict for the tunable signals only (price excluded). The
    coordinate ascent mutates these VALUES but never the KEY SET, so this is also the
    exact key-set written to data/weights.json."""
    return {s: DEFAULT_WEIGHTS[s] for s in TUNABLE}


# --------------------------------------------------------------------------- ascent

def ascend(
    objective: Callable[[dict[str, float]], float],
    seed: dict[str, float],
    *,
    metric_label: str = "tune",
    log: Callable[[str], None] = print,
) -> tuple[dict[str, float], float]:
    """Regularized coordinate ascent over TUNABLE (SPEC §6). `objective` maps a weight
    dict to a scalar to MAXIMISE (NDCG@5). Identical protocol for every pool: coarse GRID,
    strict MARGIN gate (no noise-chasing), MAX_PASSES, 0.05 floor (GRID min), price never
    entered (seed carries only TUNABLE keys). Round-preference is inherent — the grid is
    the only candidate set. Pure w.r.t. `objective`, so official and extended share it."""
    weights = dict(seed)
    best = objective(weights)
    log(f"start NDCG@5({metric_label}) = {best:.4f}  weights={weights}")
    for p in range(MAX_PASSES):
        improved = False
        for sig in TUNABLE:
            cur = weights[sig]
            for v in GRID:
                if v == cur:
                    continue
                trial = dict(weights)
                trial[sig] = v
                sc = objective(trial)
                if sc > best + MARGIN:  # strict, margin-gated: no noise-chasing
                    best, weights[sig] = sc, v
                    improved = True
        log(f"pass {p + 1}: NDCG@5({metric_label}) = {best:.4f}")
        if not improved:
            break
    return weights, best


# ------------------------------------------------------------------- pipelines / pools

def build_local_pipeline(corpus_path: Path | None = None) -> FullPipeline:
    """MEASUREMENT pipeline: provider AND mode pinned local for EVERY pool (eval integrity,
    NFR-6) — weights are never tuned against a cloud vector space, and mode=local keeps the
    LLM parse off by default (deterministic rules arm). `corpus_path` None -> official 111."""
    pois = load_pois(corpus_path) if corpus_path is not None else load_pois()
    return FullPipeline(pois, provider="local", mode="local")


def load_official_tune() -> list[EvalQuery]:
    """The 40-query official TUNE split (NEVER test — NFR-6)."""
    queries = load_eval()
    split = load_split() if SPLIT_PATH.exists() else make_split(queries)
    return select(queries, split, "tune")


def load_synth() -> list[EvalQuery]:
    """The 150 synthetic labelled queries (ground truth by construction)."""
    rows = json.loads(SYNTH_EVAL.read_text(encoding="utf-8"))
    return [EvalQuery(**r) for r in rows]


# ------------------------------------------------------- precompute + fast re-scoring
# The retrieval (intent parse + BM25 + dense + RRF relevance) and every per-POI signal
# breakdown are WEIGHT-INDEPENDENT, so we compute them ONCE per query and, for each trial
# weight vector, only re-do the linear combination + sort + the (weight-independent)
# constraint filter / anchor gate. This mirrors the pipeline's own rank_scored exactly —
# verified byte-identical to FullPipeline.rank_ids — but turns a full extended ascent from
# minutes of pipeline re-runs into seconds.

@dataclass
class _Precomp:
    intent: QueryIntent
    dense_ids: list[str]
    breakdowns: list[tuple[str, dict[str, float]]]  # (poi_id, per-signal breakdown), corpus order


def precompute(pipe: FullPipeline, query_text: str) -> _Precomp:
    intent = pipe.resolve_intent(query_text)  # mode=local -> rule parse, deterministic
    retrieval_text = intent.corrected_query or query_text
    dense_ids = [pid for pid, _ in pipe.dense.search(retrieval_text)] if pipe.dense else []
    rel = pipe._relevance(retrieval_text, intent, dense_ids)
    breakdowns = [
        (p.poi_id, pipe.ranker.signals(rel.get(p.poi_id, 0.0), intent, p,
                                       pipe._attrs[p.poi_id], pipe._review[p.poi_id]))
        for p in pipe.pois
    ]
    return _Precomp(intent, dense_ids, breakdowns)


def fast_rank_ids(pipe: FullPipeline, pc: _Precomp, weights: dict[str, float]) -> list[str]:
    """Re-score the precomputed breakdowns under `weights` and reproduce rank_scored's
    ordering + filters. `total_w = sum(weights.values())` and the SIGNALS-keyed dot product
    match LinearRanker.score exactly (price absent from `weights` -> contributes 0, as in the
    tuner); the stable sort keeps corpus-order tie-breaking identical to the pipeline."""
    total_w = sum(weights.values()) or 1.0
    scored = [(pid, sum(weights.get(k, 0.0) * b[k] for k in SIGNALS) / total_w, b)
              for pid, b in pc.breakdowns]
    scored.sort(key=lambda t: t[1], reverse=True)
    scored = pipe._constraint_filter(scored, pc.intent, pc.dense_ids)
    if pc.intent.anchor is not None:
        scored = pipe._anchor_gate(scored, pc.intent)
    return [pid for pid, _, _ in scored]


@dataclass
class QueryScorer:
    """One query bound to its home pipeline's precomputed features: weights -> ranked ids,
    plus the relevance-ordered gold ids for scoring."""
    rank: Callable[[dict[str, float]], list[str]]
    expected_ids: list[str]


def mean_ndcg5(scorers: Sequence[QueryScorer], weights: dict[str, float]) -> float:
    """Mean NDCG@5 over the (query, home-corpus) pairs — the extended objective. Equal
    weight per query (mean over pairs, matching eval.evaluate's overall mean)."""
    if not scorers:
        return 0.0
    return sum(ndcg_at_k(s.rank(weights), s.expected_ids, 5) for s in scorers) / len(scorers)


def build_scorers(pipe: FullPipeline, queries: Sequence[EvalQuery]) -> list[QueryScorer]:
    scorers: list[QueryScorer] = []
    for q in queries:
        pc = precompute(pipe, q.input_query)
        scorers.append(QueryScorer(
            rank=(lambda w, _p=pipe, _c=pc: fast_rank_ids(_p, _c, w)),
            expected_ids=q.expected_ids,
        ))
    return scorers


# ---------------------------------------------------------------------------- writers

def write_official(weights: dict[str, float], best: float) -> None:
    """Write the committed data/weights.json — exactly the 4-key structure the tuner has
    always produced (frozen; official pool only)."""
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"weights": weights, "tuned_ndcg5_tune": best, "grid": GRID, "margin": MARGIN},
                  fh, ensure_ascii=False, indent=2)


def write_candidate(payload: dict, path: Path = CANDIDATE_PATH) -> None:
    """Write the extended CANDIDATE to data/derived (gitignored). Never data/weights.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


# ------------------------------------------------------------------------------- runs

def run_official() -> None:
    """Frozen protocol: coordinate-ascend NDCG@5 on the 40-query tune split @ 111, write
    data/weights.json. Byte-identical to the historical tuner."""
    pipe = build_local_pipeline()
    tune = load_official_tune()

    def objective(weights: dict[str, float]) -> float:
        pipe.ranker.weights = weights
        return evaluate(lambda q: pipe.rank_ids(q.input_query), tune)["overall"]["ndcg@5"]

    weights, best = ascend(objective, tunable_seed(), metric_label="tune")
    write_official(weights, best)
    print(f"\nwrote {WEIGHTS_PATH}\nfinal weights={weights}")


def _eval_arm(pipe: FullPipeline, queries: Sequence[EvalQuery],
              weights: dict[str, float]) -> tuple[float, float]:
    """NDCG@5, Recall@3 for `weights` on `queries` @ pipe's corpus (naive full pipeline —
    correctness-proven, run only a handful of times for the report table)."""
    pipe.ranker.weights = weights
    o = evaluate(lambda q: pipe.rank_ids(q.input_query), queries)["overall"]
    return o["ndcg@5"], o["recall@3"]


def run_extended() -> None:
    t_start = time.perf_counter()
    off_pipe = build_local_pipeline()
    syn_pipe = build_local_pipeline(SYNTH_CORPUS)
    official_tune = load_official_tune()
    synth = load_synth()
    n_pairs = len(official_tune) + len(synth)
    print(f"extended pool: {len(official_tune)} official tune@111 + {len(synth)} synth@1000 "
          f"= {n_pairs} (query, home-corpus) pairs")

    scorers = build_scorers(off_pipe, official_tune) + build_scorers(syn_pipe, synth)
    t_precompute = time.perf_counter() - t_start

    t_asc0 = time.perf_counter()
    candidate, best = ascend(lambda w: mean_ndcg5(scorers, w), tunable_seed(),
                             metric_label="ext-190")
    t_ascent = time.perf_counter() - t_asc0

    # Committed baseline, TUNABLE-only (price excluded from BOTH arms — the table isolates the
    # 8 tuned weights, matching the ascent objective; price is a fixed non-tuned preference).
    cw = load_weights()
    committed = {k: cw[k] for k in TUNABLE}

    arms = [
        ("official tune40@111", off_pipe, official_tune),
        ("official tune40@1000", syn_pipe, official_tune),
        ("synth150@1000", syn_pipe, synth),
    ]
    table: dict[str, dict] = {}
    for label, pipe, qs in arms:
        c_ndcg, c_rec = _eval_arm(pipe, qs, committed)
        n_ndcg, n_rec = _eval_arm(pipe, qs, candidate)
        table[label] = {
            "committed": {"ndcg@5": c_ndcg, "recall@3": c_rec},
            "candidate": {"ndcg@5": n_ndcg, "recall@3": n_rec},
        }

    # Acceptance rule (evaluated, NOT enforced — orchestrator decides adoption):
    off111 = table["official tune40@111"]
    synth_arm = table["synth150@1000"]
    guard_ok = off111["candidate"]["ndcg@5"] >= off111["committed"]["ndcg@5"] - 0.005
    synth_improves = synth_arm["candidate"]["ndcg@5"] > synth_arm["committed"]["ndcg@5"]
    accept = guard_ok and synth_improves

    runtime = {"precompute_s": round(t_precompute, 1), "ascent_s": round(t_ascent, 1),
               "total_s": round(time.perf_counter() - t_start, 1)}
    payload = {
        "pool": "extended",
        "objective": "mean NDCG@5 over 190 (query, home-corpus) pairs (40 official@111 + 150 synth@1000)",
        "protocol": {"grid": GRID, "margin": MARGIN, "max_passes": MAX_PASSES,
                     "tunable": TUNABLE, "price": "fixed, excluded"},
        "candidate_weights": candidate,
        "committed_weights": committed,
        "tuned_ndcg5_ext190": best,
        "comparison": table,
        "acceptance": {"tune40@111_guard(-0.005)": guard_ok,
                       "synth150_improves": synth_improves, "accept": accept},
        "runtime_s": runtime,
    }
    write_candidate(payload)
    _print_extended_report(candidate, committed, table, best, accept, guard_ok,
                           synth_improves, runtime)


def _print_extended_report(candidate, committed, table, best, accept, guard_ok,
                           synth_improves, runtime) -> None:
    print(f"\ncandidate NDCG@5(ext-190) = {best:.4f}")
    print("\nweights (committed -> candidate):")
    for k in TUNABLE:
        c, n = committed[k], candidate[k]
        mark = "  <-- moved" if abs(c - n) > 1e-12 else ""
        print(f"  {k:11s} {c:.2f} -> {n:.2f}{mark}")

    print(f"\n{'arm':22s} {'NDCG@5 (cmt->cand)':24s} {'Recall@3 (cmt->cand)':24s}")
    for label, cell in table.items():
        cn, nn = cell["committed"]["ndcg@5"], cell["candidate"]["ndcg@5"]
        cr, nr = cell["committed"]["recall@3"], cell["candidate"]["recall@3"]
        print(f"{label:22s} {cn:.4f} -> {nn:.4f} ({nn - cn:+.4f})  "
              f"{cr:.4f} -> {nr:.4f} ({nr - cr:+.4f})")

    print(f"\nacceptance (evaluated, NOT enforced): accept={accept}")
    print(f"  tune40@111 NDCG@5 within -0.005 of committed : {guard_ok}")
    print(f"  synth150@1000 NDCG@5 improves                : {synth_improves}")
    print(f"\nruntime: precompute {runtime['precompute_s']}s + ascent {runtime['ascent_s']}s "
          f"= {runtime['total_s']}s total")
    print(f"wrote candidate -> {CANDIDATE_PATH} (gitignored; data/weights.json UNTOUCHED)")


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Tune the 8 TUNABLE signal weights (SPEC §6).")
    ap.add_argument("--pool", choices=["official", "extended"], default="official",
                    help="official (default): tune40@111 -> data/weights.json. "
                         "extended: tune40@111 + synth150@1000 -> candidate in data/derived.")
    args = ap.parse_args(argv)
    if args.pool == "official":
        run_official()
    else:
        run_extended()


if __name__ == "__main__":
    main()
