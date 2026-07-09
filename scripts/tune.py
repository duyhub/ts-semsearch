"""Weight tuning via regularized coordinate ascent (SPEC §6; eng-review A3).

Tunes the 7 signal weights on the TUNE split ONLY (never test — NFR-6), on a
COARSE grid with a minimum-improvement margin so we don't chase tune-split
noise (A3 regularization). Writes data/weights.json (committed).

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


def main() -> None:
    pois = load_pois()
    queries = load_eval()
    split = load_split() if SPLIT_PATH.exists() else make_split(queries)
    tune = select(queries, split, "tune")

    pipe = FullPipeline(pois)  # built once; we mutate ranker.weights between evals

    def score(weights: dict[str, float]) -> float:
        pipe.ranker.weights = weights
        return evaluate(lambda q: pipe.rank_ids(q.input_query), tune)["overall"]["ndcg@5"]

    weights = dict(DEFAULT_WEIGHTS)
    best = score(weights)
    print(f"start NDCG@5(tune) = {best:.4f}  weights={weights}")

    for p in range(MAX_PASSES):
        improved = False
        for sig in SIGNALS:
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
