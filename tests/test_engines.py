"""Engine-factory wiring tests (SPEC §6).

C23: `make_full_ranker`'s weight resolution used `weights or load_weights()`, which
treats an *explicit* empty dict (a falsy but intentional value) as 'unset' and
silently swaps in the tuned weights. The factory must distinguish None (unset) from
{} (explicit). Tested without the heavy dense/model path by stubbing FullPipeline.
"""
from __future__ import annotations

from semsearch import engines, pipeline
from semsearch.rank import load_weights


class _StubPipeline:
    """Records the weights it was constructed with; no model/index load."""

    def __init__(self, pois, *, weights=None, now=None, provider="local"):
        self.weights = weights

    def rank_ids(self, q):  # pragma: no cover - not exercised here
        return []


def test_explicit_empty_weights_are_not_replaced(monkeypatch):
    captured = {}

    def _factory(pois, *, weights=None, now=None, provider="local", mode=None):
        captured["weights"] = weights
        return _StubPipeline(pois, weights=weights)

    monkeypatch.setattr(pipeline, "FullPipeline", _factory)
    engines.make_full_ranker([], weights={})
    assert captured["weights"] == {}  # explicit empty dict preserved, not tuned weights


def test_none_weights_default_to_tuned(monkeypatch):
    captured = {}

    def _factory(pois, *, weights=None, now=None, provider="local", mode=None):
        captured["weights"] = weights
        return _StubPipeline(pois, weights=weights)

    monkeypatch.setattr(pipeline, "FullPipeline", _factory)
    engines.make_full_ranker([])  # weights unset (None) -> load the tuned weights
    assert captured["weights"] == load_weights()
