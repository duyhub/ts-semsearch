"""Build derived data + the committed eval split (SPEC §1, §4, §6).

  python scripts/ingest.py                 # build cache; create split if missing
  python scripts/ingest.py --rebuild-split # re-partition (changes committed split!)

Outputs:
  data/derived/pois.parquet   cache with composed doc_text (gitignored)
  data/eval_split.json        committed, stable tune/test partition
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.data import load_eval, load_pois  # noqa: E402
from semsearch.embeddings import compose_doc_text  # noqa: E402  (shared, DRY)
from semsearch.split import SPLIT_PATH, load_split, make_split, write_split  # noqa: E402

DERIVED = Path("data/derived")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-split", action="store_true",
                    help="re-partition the eval split (overwrites the committed file)")
    args = ap.parse_args()

    pois = load_pois()
    for p in pois:
        p.doc_text = compose_doc_text(p)
    DERIVED.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([p.__dict__ for p in pois])
    out = DERIVED / "pois.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out}  ({len(pois)} POIs, doc_text composed)")

    queries = load_eval()
    if SPLIT_PATH.exists() and not args.rebuild_split:
        split = load_split()
        print(f"split exists: {SPLIT_PATH}  (tune={len(split['tune'])}, test={len(split['test'])}) "
              f"— use --rebuild-split to re-partition")
    else:
        split = make_split(queries)
        write_split(split)
        action = "rebuilt" if args.rebuild_split else "created"
        print(f"{action} {SPLIT_PATH}  (seed={split['seed']}, "
              f"tune={len(split['tune'])}, test={len(split['test'])})")


if __name__ == "__main__":
    main()
