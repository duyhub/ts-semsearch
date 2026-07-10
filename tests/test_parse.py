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
    intent = parser.parse("quán cà phê yên tĩnh làm việc")
    assert intent.category == "Quán cà phê"
    assert "yên tĩnh" in intent.required_attrs
    assert "phù hợp làm việc" in intent.required_attrs
    assert intent.anchor is None


def test_wifi_near_landmark_intent(parser):
    intent = parser.parse("cafe wifi gần hồ gươm")
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


def test_district_word_boundary_no_false_match(parser):
    # "quận 10" must NOT resolve to the "Quận 1" district (substring collision).
    intent = parser.parse("quán cà phê quận 10 tphcm")
    assert intent.district != "Quận 1"


def test_subject_terms_extracted_and_block_category(parser):
    intent = parser.parse("quán bún chả khách du lịch")
    assert "bun" in intent.content_terms and "cha" in intent.content_terms
    assert intent.has_residual  # residual content present -> category filter ineligible


def test_generic_adjective_is_stopword(parser):
    # "cafe ngon": 'ngon' is a generic adjective (stopword) -> no residual content.
    intent = parser.parse("cafe ngon")
    assert intent.category == "Quán cà phê"
    assert not intent.has_residual
    assert intent.content_terms == []


def test_price_cheap_intent_parsed(parser):
    # "cafe rẻ nhất" (cheapest café): price direction = cheap; 'nhất' handled downstream.
    intent = parser.parse("cafe rẻ nhất")
    assert intent.category == "Quán cà phê"
    assert intent.price_pref == "cheap"


def test_price_expensive_intent_parsed(parser):
    intent = parser.parse("nhà hàng sang trọng")
    assert intent.category == "Nhà hàng"
    assert intent.price_pref == "expensive"


def test_price_binh_dan_is_cheap(parser):
    assert parser.parse("quán ăn bình dân").price_pref == "cheap"


def test_dat_booking_not_read_as_expensive(parser):
    # folded "đặt" (to book) collides with "đắt" (expensive) — must NOT fire price.
    intent = parser.parse("đặt bàn nhà hàng")
    assert intent.price_pref is None


def test_no_price_word_leaves_pref_none(parser):
    assert parser.parse("cafe yên tĩnh để làm việc").price_pref is None


def test_price_sang_not_fired_by_morning_sang(parser):
    # 'sáng' (morning) must NOT parse as expensive; the price key 'sang' is luxury.
    intent = parser.parse("quán ăn sáng")
    assert intent.price_pref is None


def test_category_park_not_fired_by_parking(parser):
    # 'parking' must not trip the 'park' -> Công viên keyword.
    intent = parser.parse("nhà hàng có chỗ parking")
    assert intent.category != "Công viên"


def test_tra_sua_maps_to_cafe(parser):
    assert parser.parse("trà sữa ngon").category == "Quán cà phê"


def test_bare_tra_does_not_misfire_on_son_tra(parser):
    # 'sơn trà' (a district) must not trigger the drink category.
    assert parser.parse("quán ngon sơn trà").category != "Quán cà phê"


def test_superlative_nhat_not_a_subject(parser):
    # 'nhất' is a common superlative particle -> not a distinctive subject/residual.
    intent = parser.parse("địa điểm nổi tiếng nhất")
    assert "nhat" not in intent.content_terms
    assert "nhat" not in intent.residual_terms


def test_food_subject_survives_common_word_filter(parser):
    # 'chả' folds to 'cha' but is not the common word 'cha' -> stays a subject.
    intent = parser.parse("quán bún chả khách du lịch")
    assert "bun" in intent.content_terms and "cha" in intent.content_terms


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


# --- Batch C (C3): coordinate-in-query anchors (SPEC §7.1; PRD FR-2) ---

def test_coordinate_query_sets_anchor(parser):
    intent = parser.parse("10.7738, 106.704")
    assert intent.anchor is not None
    assert intent.anchor.lat == pytest.approx(10.7738, abs=1e-4)
    assert intent.anchor.lon == pytest.approx(106.704, abs=1e-4)
    # the '.'-shattered integer shards ('10','7738',...) must not pollute the residual
    assert intent.content_terms == []
    assert not intent.has_residual
    assert intent.residual_terms == []


def test_coordinate_space_separated_parses(parser):
    intent = parser.parse("gần 10.7738 106.704")
    assert intent.anchor is not None
    assert intent.anchor.lat == pytest.approx(10.7738, abs=1e-4)
    assert intent.anchor.lon == pytest.approx(106.704, abs=1e-4)


def test_coordinate_swapped_order_parses(parser):
    intent = parser.parse("106.704, 10.7738")
    assert intent.anchor is not None
    assert intent.anchor.lat == pytest.approx(10.7738, abs=1e-4)
    assert intent.anchor.lon == pytest.approx(106.704, abs=1e-4)


def test_coordinate_with_category_parses_both(parser):
    intent = parser.parse("quán cà phê 10.7738,106.704")
    assert intent.category == "Quán cà phê"
    assert intent.anchor is not None
    assert intent.anchor.lat == pytest.approx(10.7738, abs=1e-4)
    assert not intent.content_terms  # coordinate shards are not subjects


def test_out_of_bounds_coordinate_leaves_anchor_none(parser):
    intent = parser.parse("50.0, 8.0")
    assert intent.anchor is None


def test_rating_decimal_is_not_a_coordinate(parser):
    intent = parser.parse("cà phê 3.5 sao")
    assert intent.anchor is None


def test_24_7_is_not_a_coordinate(parser):
    intent = parser.parse("cây xăng 24/7 gần đây")
    assert intent.anchor is None


def test_coordinate_takes_precedence_over_gazetteer(parser):
    # an explicit coordinate (HCMC, lat ~10.77) wins over a landmark name (Hồ Gươm,
    # Hà Nội, lat ~21.03) that also appears in the query.
    intent = parser.parse("cafe gần hồ gươm 10.7738, 106.704")
    assert intent.anchor is not None
    assert intent.anchor.lat == pytest.approx(10.7738, abs=1e-4)
