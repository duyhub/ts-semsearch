"""Evaluation harness CLI (SPEC §2, §11; PRD FR-9).

  python scripts/run_eval.py --engine random --split tune

Prints the full metric table (overall + per-difficulty + per-query_category,
each with n, plus a bootstrap CI on NDCG@5) and writes reports/metrics-*.json.

Phase 1 gate: `--engine random` prints a complete table with near-zero scores.
Later phases add --engine bm25|dense|hybrid|full.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.data import load_eval, load_pois  # noqa: E402
from semsearch.engines import (  # noqa: E402
    make_bm25_ranker,
    make_dense_ranker,
    make_hybrid_ranker,
    make_random_ranker,
)
from semsearch.eval import METRIC_KEYS, evaluate  # noqa: E402
from semsearch.split import SPLIT_PATH, load_split, make_split, select  # noqa: E402

REPORTS = Path("reports")


def build_engine(name: str, pois):
    if name == "random":
        return make_random_ranker(pois, seed=0)
    if name == "bm25":
        return make_bm25_ranker(pois)
    if name == "dense":
        return make_dense_ranker(pois)
    if name == "hybrid":
        return make_hybrid_ranker(pois)
    raise SystemExit(f"unknown engine {name!r} (supported: random, bm25, dense, hybrid)")


def _row(label: str, cell: dict) -> str:
    vals = "  ".join(f"{cell[k]:>7.3f}" for k in METRIC_KEYS)
    n = f"(n={cell['n']})" if "n" in cell else ""
    return f"  {label:<26}{n:<8}{vals}"


def print_report(result: dict, engine: str, split: str) -> None:
    print(f"\nENGINE={engine}  SPLIT={split}  n={result['n']}")
    header = "  " + " " * 34 + "  ".join(f"{k:>7}" for k in METRIC_KEYS)
    print(header)
    ov = {**result["overall"], "n": result["n"]}
    ci = result["ci"]
    print(_row("overall", ov) + f"   [{ci['metric']} 95% CI {ci['lo']:.3f}-{ci['hi']:.3f}]")
    print("  by difficulty")
    for k in ("Hard", "Medium", "Easy"):
        if k in result["by_difficulty"]:
            print(_row("  " + k, result["by_difficulty"][k]))
    print("  by query_category")
    for k, cell in sorted(result["by_category"].items(), key=lambda kv: -kv[1]["n"]):
        note = "  <- n=1, anecdotal" if cell["n"] == 1 else ""
        print(_row("  " + k, cell) + note)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="random")
    ap.add_argument("--split", default="tune", choices=["tune", "test", "all"])
    args = ap.parse_args()

    pois = load_pois()
    queries = load_eval()
    if args.split != "all":
        split = load_split() if SPLIT_PATH.exists() else make_split(queries)
        queries = select(queries, split, args.split)

    engine = build_engine(args.engine, pois)
    result = evaluate(engine, queries)
    print_report(result, args.engine, args.split)

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"metrics-{args.engine}-{args.split}.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"engine": args.engine, "split": args.split, **result}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
