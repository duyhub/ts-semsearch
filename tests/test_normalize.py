"""Folding, abbreviation expansion, and the fuzzy canonicalize guard (SPEC §3, §11)."""
from __future__ import annotations

from semsearch.normalize import canonicalize, doc_tokens, expand_query, fold, tokenize


def test_fold_strips_diacritics_and_case():
    assert fold("Quán Cà Phê") == "quan ca phe"
    assert fold("Đà Nẵng") == "da nang"
    assert fold("Hồ Gươm") == "ho guom"


def test_fold_preserves_24_7_token():
    assert tokenize("mở 24/7 gần đây") == ["mo", "24/7", "gan", "day"]


def test_expand_abbreviations_and_qnumber():
    # "cf" -> "ca phe", "q1" -> "quan 1"
    assert expand_query("cf q1 co wifi") == ["ca", "phe", "quan", "1", "co", "wifi"]
    assert expand_query("ks gan bien") == ["khach", "san", "gan", "bien"]
    assert expand_query("quan cafe yen tinh gan hcm") == [
        "quan", "ca", "phe", "yen", "tinh", "gan", "tp", "hcm",
    ]


def test_expand_bigram_abbreviation():
    assert expand_query("gas station 24/7") == ["tram", "xang", "24/7"]


def test_doc_tokens_fold_only():
    assert doc_tokens("The Workshop Coffee") == ["the", "workshop", "coffee"]


def test_canonicalize_exact_and_unique_fuzzy():
    vocab = ["wifi", "cafe", "quan"]
    assert canonicalize("wifi", vocab) == "wifi"          # exact
    assert canonicalize("wifl", vocab) == "wifi"          # edit 1, len>=4, unique


def test_canonicalize_refuses_ambiguous_and_short():
    # "bane" is edit-1 from both "band" and "cane" -> refuse (C2 precision guard)
    assert canonicalize("bane", ["band", "cane"]) is None
    # too short to fuzzy-match
    assert canonicalize("caf", ["cafe"]) is None
    # no candidate within edit 1
    assert canonicalize("zzzz", ["wifi", "cafe"]) is None
