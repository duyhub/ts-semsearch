"""Explanation generation + faithfulness validator (SPEC §8, §11)."""
from __future__ import annotations

from semsearch.data import POI, Anchor, QueryIntent
from semsearch.explain import generate_reasons, validate_reasons


def _poi(**kw) -> POI:
    base = dict(
        poi_id="C001", name="The Workshop Coffee", brand="The Workshop", category="Quán cà phê",
        sub_category=None, city="TP.HCM", district="Quận 1", address="", lat=10.7738, lon=106.704,
        rating=4.6, review_count=1560, popularity=91, price_level=3,
        opening_hours="07:00-22:30", attributes=["wifi", "yên tĩnh", "phù hợp làm việc"],
        tags=[], description="",
    )
    base.update(kw)
    return POI(**base)


def test_generated_reasons_are_faithful():
    intent = QueryIntent(raw="", normalized="", required_attrs=["wifi", "yên tĩnh"],
                         anchor=Anchor("Hồ Gươm", 10.7750, 106.7000))
    poi = _poi()
    reasons = generate_reasons(intent, poi)
    assert 1 <= len(reasons) <= 4
    assert any("wifi" in r and "yên tĩnh" in r for r in reasons)     # matched attrs
    assert any("4.6★" in r and "1.560 đánh giá" in r for r in reasons)  # true rating/reviews
    assert validate_reasons(reasons, poi) == []                      # faithful by construction


def test_validator_rejects_fabricated_attribute():
    poi = _poi()
    bad = ["✓ hồ bơi, ✓ wifi"]  # hồ bơi is NOT on the POI
    violations = validate_reasons(bad, poi)
    assert violations
    assert any("hồ bơi" in v for v in violations)


def test_validator_rejects_wrong_numbers():
    poi = _poi(rating=4.6, review_count=1560)
    assert validate_reasons(["5.0★ · 1.560 đánh giá"], poi)   # rating fabricated
    assert validate_reasons(["4.6★ · 9.999 đánh giá"], poi)   # review_count fabricated
    assert validate_reasons(["4.6★ · 1.560 đánh giá"], poi) == []  # both correct


def test_reasons_capped_at_max():
    intent = QueryIntent(raw="", normalized="", required_attrs=["wifi"],
                         anchor=Anchor("Hồ Gươm", 10.77, 106.70))
    assert len(generate_reasons(intent, _poi(), max_reasons=2)) == 2
