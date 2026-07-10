"""Weight tuning via regularized coordinate ascent (SPEC §6; eng-review A3, C4).

Tunes the 8 TUNABLE signal weights on the TUNE split ONLY (never test — NFR-6), on
a COARSE grid with a minimum-improvement margin so we don't chase tune-split noise
(A3 regularization). Writes data/weights.json (committed) with exactly those 8 keys.

`price` is DELIBERATELY EXCLUDED (C4): it is a fixed 0.20 affordability-preference
weight, never eval-tuned (only 2/60 eval queries express price — too few to inform
it, NFR-6). It never enters the working weight dict here, so it is never scored,
never coordinate-ascended, and never written; load_weights() re-supplies it from
DEFAULT_WEIGHTS via its per-key fallback.

  python scripts/tune.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.data import load_eval, load_pois  # noqa: E402
from semsearch.eval import evaluate  # noqa: E402
from semsearch.pipeline import FullPipeline  # noqa: E402
from semsearch.rank import DEFAULT_WEIGHTS, SIGNALS, WEIGHTS_PATH  # noqa: E402
from semsearch.split import SPLIT_PATH, load_split, make_split, select  # noqa: E402

GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # coarse (A3); 0.05 floor keeps every signal live (FR-7)
MARGIN = 1e-3  # require a real improvement before moving a weight (regularization)
MAX_PASSES = 2

# The signals we actually tune: everything except the fixed-weight `price` (C4).
TUNABLE = [s for s in SIGNALS if s != "price"]


def tunable_seed() -> dict[str, float]:
    """Pre-tuning weight dict for the tunable signals only (price excluded). The
    coordinate ascent mutates these VALUES but never the KEY SET, so this is also the
    exact key-set written to data/weights.json."""
    return {s: DEFAULT_WEIGHTS[s] for s in TUNABLE}


def main() -> None:
    pois = load_pois()
    queries = load_eval()
    split = load_split() if SPLIT_PATH.exists() else make_split(queries)
    tune = select(queries, split, "tune")

    pipe = FullPipeline(pois)  # built once; we mutate ranker.weights between evals

    def score(weights: dict[str, float]) -> float:
        pipe.ranker.weights = weights
        return evaluate(lambda q: pipe.rank_ids(q.input_query), tune)["overall"]["ndcg@5"]

    weights = tunable_seed()  # 8 tunable signals; price is fixed and never entered here (C4)
    best = score(weights)
    print(f"start NDCG@5(tune) = {best:.4f}  weights={weights}")

    for p in range(MAX_PASSES):
        improved = False
        for sig in TUNABLE:
            cur = weights[sig]
            for v in GRID:
                if v == cur:
                    continue
                trial = dict(weights)
                trial[sig] = v
                sc = score(trial)
                if sc > best + MARGIN:  # strict, margin-gated: no noise-chasing
                    best, weights[sig] = sc, v
                    improved = True
            pipe.ranker.weights = weights  # reset to current best for next signal
        print(f"pass {p + 1}: NDCG@5(tune) = {best:.4f}")
        if not improved:
            break

    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(WEIGHTS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"weights": weights, "tuned_ndcg5_tune": best, "grid": GRID, "margin": MARGIN},
                  fh, ensure_ascii=False, indent=2)
    print(f"\nwrote {WEIGHTS_PATH}\nfinal weights={weights}")


if __name__ == "__main__":
    main()
