"""Metric math against hand-computed fixtures (SPEC §11)."""
from __future__ import annotations

import math

import pytest

from semsearch.eval import bootstrap_ci, mrr, ndcg_at_k, recall_at_k

# Worked example (hand-computed):
#   expected order ['A','B','C'] -> gains A=3, B=2, C=1
#   ranked ['B','X','A','C'], X irrelevant, k=5
EXPECTED = ["A", "B", "C"]
RANKED = ["B", "X", "A", "C"]


def test_recall_at_k():
    # top-3 = {B,X,A}; relevant = {A,B,C} -> 2/3
    assert recall_at_k(RANKED, EXPECTED, 3) == pytest.approx(2 / 3)
    # top-5 covers all 4 ranked; hits {A,B,C} -> 3/3
    assert recall_at_k(RANKED, EXPECTED, 5) == pytest.approx(1.0)


def test_recall_edges():
    assert recall_at_k([], EXPECTED, 5) == 0.0
    assert recall_at_k(RANKED, [], 5) == 0.0
    assert recall_at_k(["X", "Y"], EXPECTED, 5) == 0.0


def test_ndcg_worked_example():
    # DCG = 2/log2(2) + 3/log2(4) + 1/log2(5) = 2 + 1.5 + 0.43068 = 3.93068
    dcg = 2 / math.log2(2) + 3 / math.log2(4) + 1 / math.log2(5)
    # IDCG (ideal [3,2,1]) = 3 + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5 = 4.76186
    idcg = 3 + 2 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, EXPECTED, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(RANKED, EXPECTED, 5) == pytest.approx(0.82545, abs=1e-4)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["A", "B", "C"], EXPECTED, 5) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, [], 5) == 0.0
    assert ndcg_at_k([], EXPECTED, 5) == 0.0


def test_mrr():
    assert mrr(RANKED, EXPECTED) == pytest.approx(1.0)  # B at rank 1
    assert mrr(["X", "Y", "C"], EXPECTED) == pytest.approx(1 / 3)  # C at rank 3
    assert mrr(["X", "Y"], EXPECTED) == 0.0


def test_bootstrap_ci_deterministic_and_bracketing():
    vals = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]  # mean 0.5
    lo1, hi1 = bootstrap_ci(vals)
    lo2, hi2 = bootstrap_ci(vals)
    assert (lo1, hi1) == (lo2, hi2)  # fixed seed -> reproducible
    assert lo1 <= 0.5 <= hi1
    # degenerate inputs
    assert bootstrap_ci([]) == (0.0, 0.0)
    assert bootstrap_ci([0.7]) == (0.7, 0.7)
