"""Haversine + anchor resolution (SPEC §7, §11)."""
from __future__ import annotations

import pytest

from semsearch.data import load_pois
from semsearch.geo import COORD_ANCHOR_NAME, Gazetteer, detect_coordinate_anchor, haversine


def test_haversine_zero_and_known():
    assert haversine(21.0287, 105.8524, 21.0287, 105.8524) == pytest.approx(0.0, abs=1e-6)
    # Hồ Gươm -> Hồ Tây is roughly 3-4 km
    d = haversine(21.0287, 105.8524, 21.0587, 105.8190)
    assert 3.0 < d < 5.0


@pytest.fixture(scope="module")
def gaz():
    return Gazetteer(load_pois())


def test_resolve_landmark(gaz):
    a = gaz.resolve("cafe co wifi gan ho guom")
    assert a is not None
    assert a.lat == pytest.approx(21.0287, abs=0.01)


def test_resolve_district_centroid(gaz):
    a = gaz.resolve("quan cafe yen tinh o quan 1 tp hcm")
    assert a is not None  # resolves via district centroid or a POI in Quận 1


def test_resolve_none_for_unknown(gaz):
    assert gaz.resolve("noi nao do khong ten") is None


def test_district_centroid_key_exists_for_quan_1(gaz):
    # Fix B relies on the gazetteer holding a folded "quan 1" district centroid
    # so an expanded "q1" -> "quan 1" resolves to it.
    assert "quan 1" in gaz.districts


# --- Fix 1: diacritic-compatible landmark resolution ---

def test_landmark_no_false_fire_on_pho_co(gaz):
    # 'phở có chỗ ...' must NOT anchor to Phố Cổ (Hanoi) for a phở query.
    assert gaz.resolve("quan pho co cho ngoi ngoai troi",
                       "quán phở có chỗ ngồi ngoài trời") is None


def test_landmark_pho_co_diacritic_forms(gaz):
    # exact 'phố cổ' anchors; 'phở cổ' does not (ở ≠ ố); unaccented 'pho co' does.
    assert gaz.resolve("pho co", "phố cổ") is not None
    assert gaz.resolve("pho co", "phở cổ") is None
    a = gaz.resolve("cafe gan pho co", "cafe gan pho co")
    assert a is not None and a.name == "Phố Cổ"


def test_landmark_ho_tay_not_in_cho_tay(gaz):
    assert gaz.resolve("tim cho tay ba lo", "tìm chỗ tây ba lô") is None


# --- Batch C (C3): coordinate-in-query anchor detection (SPEC §7.1; PRD FR-2) ---

def test_detect_coordinate_basic():
    a = detect_coordinate_anchor("10.7738, 106.704")
    assert a is not None
    assert a.name == COORD_ANCHOR_NAME
    assert a.lat == pytest.approx(10.7738)
    assert a.lon == pytest.approx(106.704)


def test_detect_coordinate_space_separated():
    a = detect_coordinate_anchor("gần 10.7738 106.704")
    assert a is not None
    assert a.lat == pytest.approx(10.7738)
    assert a.lon == pytest.approx(106.704)


def test_detect_coordinate_no_space_after_comma():
    a = detect_coordinate_anchor("10.7738,106.704")
    assert a is not None
    assert (a.lat, a.lon) == pytest.approx((10.7738, 106.704))


def test_detect_coordinate_swapped_order():
    # users paste (lon, lat) too; swap only when the natural order fails bounds.
    a = detect_coordinate_anchor("106.704, 10.7738")
    assert a is not None
    assert a.lat == pytest.approx(10.7738)
    assert a.lon == pytest.approx(106.704)


def test_detect_coordinate_out_of_vn_bounds_none():
    # Frankfurt (50.0, 8.0): out of Vietnam bounds both ways -> no anchor.
    assert detect_coordinate_anchor("50.0, 8.0") is None


def test_detect_coordinate_single_decimal_none():
    # a lone rating decimal is not a coordinate pair.
    assert detect_coordinate_anchor("cà phê 3.5 sao") is None


def test_detect_coordinate_24_7_not_a_pair():
    assert detect_coordinate_anchor("cây xăng 24/7 gần đây") is None


def test_detect_coordinate_hyphen_separator_not_a_pair():
    # a bare hyphen is a range, not a coordinate separator.
    assert detect_coordinate_anchor("10.7738-106.704") is None


def test_detect_coordinate_embedded_in_text():
    a = detect_coordinate_anchor("quán cà phê 10.7738, 106.704 gần đây")
    assert a is not None
    assert a.lat == pytest.approx(10.7738)
