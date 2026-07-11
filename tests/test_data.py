"""Loader/coercion tests (SPEC §2). Guards the xlsx→typed-frame parsing helpers.

C20: a blank xlsx cell arrives as float('nan'), which is *truthy* — so a naive
`str(cell or "")` turns an empty description into the literal string 'nan'. These
tests pin the NaN-safe coercion (via `_opt_str`) plus the documented behaviour of
the other small loaders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from semsearch import data as D
from semsearch.data import _opt_int, _opt_str, _split, load_pois


def _raw_poi_row(**over) -> dict:
    """A minimal raw POI_Dataset row (pre-rename column names)."""
    row = dict(
        poi_id="C001", poi_name="Café X", category="Quán cà phê", city="TP.HCM",
        district="Quận 1", address="1 Lê Lợi", latitude=10.77, longitude=106.70,
        rating=4.5, review_count=1000, popularity_score=80.0,
        brand="X", sub_category="cà phê", price_level=2, opening_hours="07:00-22:30",
        attributes="wifi;yên tĩnh", tags="lãng mạn", description="Quán đẹp",
    )
    row.update(over)
    return row


def test_nan_description_becomes_empty_string(monkeypatch):
    # C20: a blank description cell is NaN (truthy) -> must coerce to '' not 'nan'.
    frame = pd.DataFrame([_raw_poi_row(description=float("nan"))])
    monkeypatch.setattr(D.pd, "read_excel", lambda *a, **k: frame)
    pois = load_pois()
    assert len(pois) == 1
    assert pois[0].description == ""


def test_present_description_is_preserved(monkeypatch):
    frame = pd.DataFrame([_raw_poi_row(description="  Quán yên tĩnh  ")])
    monkeypatch.setattr(D.pd, "read_excel", lambda *a, **k: frame)
    assert load_pois()[0].description == "Quán yên tĩnh"  # stripped, preserved


def test_opt_str_nan_and_blank_are_none():
    assert _opt_str(float("nan")) is None
    assert _opt_str("   ") is None
    assert _opt_str(None) is None
    assert _opt_str("  x ") == "x"


def test_opt_int_truncates_and_handles_nan():
    # documents int() truncation-toward-zero and NaN/None -> None
    assert _opt_int(4.0) == 4
    assert _opt_int(4.7) == 4          # truncates, does not round
    assert _opt_int(np.float64(3.0)) == 3
    assert _opt_int(float("nan")) is None
    assert _opt_int(None) is None


def test_split_trims_blanks_and_whitespace():
    assert _split("a; b ;;  c ", ";") == ["a", "b", "c"]
    assert _split(float("nan"), ";") == []
    assert _split(None, ";") == []
    assert _split("", ";") == []
