"""Rule parser golden tests (SPEC §7, §11)."""
from __future__ import annotations

import pytest

from semsearch.data import load_pois
from semsearch.geo import Gazetteer
from semsearch.parse import Parser


@pytest.fixture(scope="module")
def parser():
    pois = load_pois()
    return Parser(pois, Gazetteer(pois))


def test_work_cafe_intent(parser):
    intent = parser.parse("quán cà phê yên tĩnh để làm việc")
    assert intent.category == "Quán cà phê"
    assert "yên tĩnh" in intent.required_attrs
    assert "phù hợp làm việc" in intent.required_attrs
    assert intent.anchor is None


def test_wifi_near_landmark_intent(parser):
    intent = parser.parse("cafe có wifi gần hồ gươm")
    assert intent.category == "Quán cà phê"          # cafe -> cà phê
    assert "wifi" in intent.required_attrs
    assert intent.anchor is not None
    assert intent.anchor.lat == pytest.approx(21.0287, abs=0.01)


def test_gas_24_7_intent(parser):
    intent = parser.parse("cây xăng 24/7 gần đây")
    assert intent.category == "Trạm xăng"
    assert "24/7" in intent.required_attrs


def test_non_accented_still_parses(parser):
    intent = parser.parse("quan cafe yen tinh")
    assert intent.category == "Quán cà phê"
    assert "yên tĩnh" in intent.required_attrs


def test_abbrev_district_resolves_anchor_and_district(parser):
    # "q1 tphcm" (abbreviated) must resolve to the Quận 1 district anchor and
    # populate intent.district — previously the folded "q1" never matched the
    # gazetteer key "quan 1", leaving anchor None (the reported bug).
    intent = parser.parse("quan ca phe o q1 tphcm")
    assert intent.category == "Quán cà phê"
    assert intent.city == "TP.HCM"
    assert intent.anchor is not None
    assert intent.anchor.lat == pytest.approx(10.77, abs=0.05)  # Quận 1 centroid
    assert intent.district is not None
