"""Haversine + anchor resolution (SPEC §7, §11)."""
from __future__ import annotations

import pytest

from semsearch.data import load_pois
from semsearch.geo import Gazetteer, haversine


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
