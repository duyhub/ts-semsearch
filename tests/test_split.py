"""Split stratification + determinism (SPEC §6)."""
from __future__ import annotations

from collections import Counter

import pytest

from semsearch.data import load_eval
from semsearch.split import DEFAULT_SEED, make_split


@pytest.fixture(scope="module")
def queries():
    return load_eval()


def test_split_is_deterministic(queries):
    a = make_split(queries)
    b = make_split(queries)
    assert a == b


def test_split_sizes_40_20(queries):
    split = make_split(queries)
    assert len(split["tune"]) == 40
    assert len(split["test"]) == 20


def test_split_disjoint_and_covers_all(queries):
    split = make_split(queries)
    tune, test = set(split["tune"]), set(split["test"])
    assert tune.isdisjoint(test)
    all_ids = {q.query_id for q in queries}
    assert tune | test == all_ids


def test_split_stratified_by_difficulty(queries):
    split = make_split(queries)
    diff = {q.query_id: q.difficulty for q in queries}
    test_by_diff = Counter(diff[qid] for qid in split["test"])
    # 25 Hard / 30 Medium / 5 Easy, test_frac 1/3 -> round(): 8 / 10 / 2
    assert test_by_diff["Hard"] == 8
    assert test_by_diff["Medium"] == 10
    assert test_by_diff["Easy"] == 2


def test_seed_change_repartitions(queries):
    base = make_split(queries)
    other = make_split(queries, seed=DEFAULT_SEED + 1)
    assert base["test"] != other["test"]
