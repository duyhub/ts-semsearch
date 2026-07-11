"""7-signal ranker unit tests (SPEC §6, §11) — the core ML deliverable."""
from __future__ import annotations

from datetime import datetime

import pytest

from semsearch.data import POI, Anchor, QueryIntent
from semsearch.rank import (
    LinearRanker,
    attributes_signal,
    category_signal,
    distance_signal,
    open_now_signal,
    popularity_signal,
    price_signal,
    rating_signal,
    semantic_signal,
)


def _poi(**kw) -> POI:
    base = dict(
        poi_id="C001", name="X", brand=None, category="Quán cà phê", sub_category=None,
        city="TP.HCM", district="Quận 1", address="", lat=10.77, lon=106.70,
        rating=4.5, review_count=1000, popularity=80, price_level=2,
        opening_hours="07:00-22:30", attributes=["wifi", "yên tĩnh"], tags=[], description="",
    )
    base.update(kw)
    return POI(**base)


def _intent(**kw) -> QueryIntent:
    base = dict(raw="", normalized="")
    base.update(kw)
    return QueryIntent(**base)


def test_semantic_fixed_band_is_candidate_set_independent():
    # OV6: a fixed calibrated band, not per-query min-max
    assert semantic_signal(0.30) == pytest.approx(0.0)
    assert semantic_signal(0.75) == pytest.approx(1.0)
    assert semantic_signal(0.525) == pytest.approx(0.5)
    assert semantic_signal(0.10) == 0.0   # clamp low
    assert semantic_signal(0.90) == 1.0   # clamp high


def test_attributes_signal():
    intent = _intent(required_attrs=["wifi", "yên tĩnh"])
    assert attributes_signal(intent, {"wifi", "yen tinh"}) == pytest.approx(1.0)
    assert attributes_signal(intent, {"wifi"}) == pytest.approx(0.5)
    assert attributes_signal(_intent(), {"wifi"}) == 0.5  # no required -> neutral


def test_category_signal():
    # neutral (inert) when the query has no parsed category
    assert category_signal(_intent(), _poi(category="Quán cà phê")) == 0.5
    # 1.0 on exact category match, 0.0 on mismatch
    assert category_signal(_intent(category="Quán cà phê"), _poi(category="Quán cà phê")) == 1.0
    assert category_signal(_intent(category="Quán cà phê"), _poi(category="Trạm xăng")) == 0.0


def test_distance_signal_neutral_without_anchor_and_one_at_zero():
    assert distance_signal(_intent(), _poi()) == 0.5
    p = _poi(lat=10.77, lon=106.70)
    intent = _intent(anchor=Anchor("x", 10.77, 106.70))
    assert distance_signal(intent, p) == pytest.approx(1.0, abs=1e-6)


def test_distance_weight_is_excluded_without_a_real_anchor():
    ranker = LinearRanker(
        weights={"semantic": 1.0, "distance": 1.0},
        now=datetime(2026, 7, 11, 14, 0),
        global_rating_mean=4.2,
    )
    score, breakdown = ranker.score(1.0, _intent(), _poi(), set(), set())
    assert breakdown["distance"] == 0.5  # diagnostic value remains explicit
    assert score == pytest.approx(1.0)  # distance weight is absent, not averaged as 0.5

    anchored_score, _ = ranker.score(
        0.0,
        _intent(anchor=Anchor("x", 10.77, 106.70)),
        _poi(),
        set(),
        set(),
    )
    assert anchored_score == pytest.approx(0.5)  # (semantic 0 + distance 1) / 2


def test_rating_signal_monotonic():
    high = rating_signal(_poi(rating=4.7, review_count=5000), 4.2)
    low = rating_signal(_poi(rating=3.8, review_count=150), 4.2)
    assert high > low
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


def test_popularity_signal():
    assert popularity_signal(_poi(popularity=90)) == pytest.approx(0.9)


def test_open_now_three_cases_and_determinism():
    afternoon = datetime(2026, 7, 11, 14, 0)
    night = datetime(2026, 7, 11, 2, 0)
    # 24/7 always open
    assert open_now_signal(_intent(), _poi(opening_hours="24/7"), afternoon) == 1.0
    # standard hours: open at 14:00, closed at 02:00
    assert open_now_signal(_intent(), _poi(opening_hours="07:00-22:30"), afternoon) == 1.0
    assert open_now_signal(_intent(), _poi(opening_hours="07:00-22:30"), night) == 0.3
    # overnight wraparound 18:00-03:00: closed at 14:00, open at 02:00
    assert open_now_signal(_intent(), _poi(opening_hours="18:00-03:00"), afternoon) == 0.3
    assert open_now_signal(_intent(), _poi(opening_hours="18:00-03:00"), night) == 1.0
    # unknown hours -> neutral
    assert open_now_signal(_intent(), _poi(opening_hours=None), afternoon) == 0.5


def test_open_now_determinism_across_times():
    # same POI + same injected now -> identical (A1 reproducibility)
    p = _poi(opening_hours="18:00-03:00")
    t = datetime(2026, 7, 11, 20, 0)
    assert open_now_signal(_intent(), p, t) == open_now_signal(_intent(), p, t)


def test_price_signal_directions():
    cheap = _intent(price_pref="cheap")
    pricey = _intent(price_pref="expensive")
    # cheap intent: level 1 (cheapest) scores high, level 4 (priciest) scores low
    assert price_signal(cheap, _poi(price_level=1)) == pytest.approx(1.0)
    assert price_signal(cheap, _poi(price_level=4)) == pytest.approx(0.0)
    assert price_signal(cheap, _poi(price_level=1)) > price_signal(cheap, _poi(price_level=3))
    # expensive intent inverts
    assert price_signal(pricey, _poi(price_level=4)) == pytest.approx(1.0)
    assert price_signal(pricey, _poi(price_level=1)) == pytest.approx(0.0)
    # no price intent -> neutral (constant across POIs -> no ranking effect)
    assert price_signal(_intent(), _poi(price_level=1)) == 0.5
    assert price_signal(_intent(), _poi(price_level=4)) == 0.5
    # unknown price_level -> neutral even with an intent
    assert price_signal(cheap, _poi(price_level=None)) == 0.5
