"""Tune/test split (SPEC §6, PRD NFR-6).

Stratified 40/20 by difficulty, fixed seed, committed to the repo. The test
split must NEVER influence code, weights, or vocabularies — it is evaluated once
per milestone. Keeping the split here (pure, seeded, deterministic) lets a test
assert that property (tests/test_integrity.py).

The committed split lives at data/eval_split.json (NOT data/derived/, which is
gitignored) so it is version-controlled and reproducible from a fresh clone.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from .data import EvalQuery

SPLIT_PATH = Path("data/eval_split.json")
DEFAULT_SEED = 20260712  # committed seed; changing it re-partitions the eval set
TEST_FRAC = 1 / 3  # 60 queries -> 40 tune / 20 test


def make_split(
    queries: Sequence[EvalQuery], *, seed: int = DEFAULT_SEED, test_frac: float = TEST_FRAC
) -> dict:
    """Deterministic stratified-by-difficulty split. Returns tune/test id lists.

    Uses stdlib random.Random(seed) (stable Mersenne Twister) so the partition is
    identical across machines and Python 3.11+ runs.
    """
    by_diff: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        by_diff[q.difficulty].append(q.query_id)

    tune: list[str] = []
    test: list[str] = []
    for difficulty in sorted(by_diff):  # deterministic group order
        ids = sorted(by_diff[difficulty])  # deterministic base order before shuffle
        random.Random(f"{seed}:{difficulty}").shuffle(ids)
        n_test = round(len(ids) * test_frac)
        test.extend(ids[:n_test])
        tune.extend(ids[n_test:])

    return {
        "seed": seed,
        "test_frac": test_frac,
        "tune": sorted(tune),
        "test": sorted(test),
    }


def load_split(path: str | Path = SPLIT_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_split(split: dict, path: str | Path = SPLIT_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(split, fh, ensure_ascii=False, indent=2)


def select(queries: Sequence[EvalQuery], split: dict, which: str) -> list[EvalQuery]:
    """Return the queries belonging to the named split ('tune' or 'test')."""
    if which not in ("tune", "test"):
        raise ValueError(f"split must be 'tune' or 'test', got {which!r}")
    wanted = set(split[which])
    return [q for q in queries if q.query_id in wanted]
