"""IR metrics + aggregation (SPEC §2, §6; PRD FR-9).

Metrics are the requirement of record. NDCG uses graded relevance: the first
expected id has gain 3, the second 2, the rest 1 (SPEC §2). Recall@k and MRR
use the full expected set as the relevant set. Aggregation breaks results down
per-difficulty AND per-query_category with n counts, and reports a bootstrap
confidence interval so a 20-query test number is never a bare point estimate
(eng-review A3).

Determinism (NFR-5/7): the bootstrap uses a fixed seed; no wall-clock, no
global RNG state.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from .data import EvalQuery

# A ranking engine maps a query to a relevance-ordered list of poi_ids.
RankFn = Callable[[EvalQuery], list[str]]


def _gains(expected_ordered: Sequence[str]) -> dict[str, int]:
    """Graded gain per expected id: 1st -> 3, 2nd -> 2, rest -> 1 (SPEC §2)."""
    gains: dict[str, int] = {}
    for i, pid in enumerate(expected_ordered):
        gains[pid] = 3 if i == 0 else 2 if i == 1 else 1
    return gains


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hits = sum(1 for pid in ranked_ids[:k] if pid in relevant)
    return hits / len(relevant)


def ndcg_at_k(ranked_ids: Sequence[str], expected_ordered: Sequence[str], k: int) -> float:
    gains = _gains(expected_ordered)
    if not gains:
        return 0.0
    dcg = 0.0
    for rank, pid in enumerate(ranked_ids[:k]):
        g = gains.get(pid, 0)
        if g:
            dcg += g / math.log2(rank + 2)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def per_query_metrics(ranked_ids: Sequence[str], q: EvalQuery) -> dict[str, float]:
    return {
        "recall@3": recall_at_k(ranked_ids, q.expected_ids, 3),
        "recall@5": recall_at_k(ranked_ids, q.expected_ids, 5),
        "ndcg@5": ndcg_at_k(ranked_ids, q.expected_ids, 5),
        "mrr": mrr(ranked_ids, q.expected_ids),
    }


METRIC_KEYS = ("recall@3", "recall@5", "ndcg@5", "mrr")


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 20260712
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Deterministic (fixed seed).

    Small test splits (20 queries) make a point estimate misleading; this gives
    the interval the headline metric should carry (eng-review A3).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


@dataclass
class MetricCell:
    n: int
    means: dict[str, float]


def _aggregate(rows: list[dict[str, float]]) -> MetricCell:
    n = len(rows)
    if n == 0:
        return MetricCell(0, {k: 0.0 for k in METRIC_KEYS})
    means = {k: sum(r[k] for r in rows) / n for k in METRIC_KEYS}
    return MetricCell(n, means)


def evaluate(rank_fn: RankFn, queries: Sequence[EvalQuery], *, ci_metric: str = "ndcg@5") -> dict:
    """Run the engine over every query and aggregate.

    Returns overall means, per-difficulty and per-query_category cells (each with
    its n), a bootstrap CI on `ci_metric`, and the raw per-query rows.
    """
    per_query: list[dict] = []
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    by_category: dict[str, list[dict]] = defaultdict(list)

    for q in queries:
        ranked = rank_fn(q)
        m = per_query_metrics(ranked, q)
        row = {"query_id": q.query_id, **m}
        per_query.append(row)
        by_difficulty[q.difficulty].append(m)
        by_category[q.query_category].append(m)

    overall = _aggregate([{k: r[k] for k in METRIC_KEYS} for r in per_query])
    ci = bootstrap_ci([r[ci_metric] for r in per_query]) if per_query else (0.0, 0.0)

    return {
        "n": overall.n,
        "overall": overall.means,
        "ci": {"metric": ci_metric, "lo": ci[0], "hi": ci[1]},
        "by_difficulty": {k: {"n": c.n, **c.means} for k, c in ((d, _aggregate(v)) for d, v in by_difficulty.items())},
        "by_category": {k: {"n": c.n, **c.means} for k, c in ((d, _aggregate(v)) for d, v in by_category.items())},
        "per_query": per_query,
    }
