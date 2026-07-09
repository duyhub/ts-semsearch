"""Full-pipeline behavior tests (SPEC §5-6)."""
from __future__ import annotations

import pytest

from semsearch.data import load_pois
from semsearch.pipeline import FullPipeline


@pytest.fixture(scope="module")
def pipe():
    return FullPipeline(load_pois())


def test_anchor_gate_keeps_near_anchor_on_top(pipe):
    # "gần hồ gươm" resolves to a Hà Nội anchor; the top results must be near it,
    # never an other-city café 600+ km away (the Phase-9 distance-gate fix).
    _, results = pipe.search("cafe có wifi gần hồ gươm", k=5)
    assert results
    assert all(r.poi.city == "Hà Nội" for r in results[:3])


def test_no_anchor_query_unaffected(pipe):
    # a query with no location anchor still returns quiet work cafés on top
    _, results = pipe.search("quán cà phê yên tĩnh để làm việc", k=3)
    assert all(r.poi.category == "Quán cà phê" for r in results)


def test_gibberish_still_nonempty(pipe):
    assert pipe.rank_ids("zzzz qwerty asdf")  # full-corpus ranking never empty
