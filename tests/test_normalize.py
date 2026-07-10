"""Folding, abbreviation expansion, and the fuzzy canonicalize guard (SPEC §3, §11)."""
from __future__ import annotations

from semsearch.normalize import (
    canonicalize,
    compat_token_seq,
    doc_tokens,
    expand_query,
    fold,
    query_common_tokens,
    token_key_matches,
    tokenize,
)


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


# --- Fix 1: diacritic-compatible, token-boundary matching (compat_token_seq) ---

def test_compat_token_seq_diacritic_conflict_rejects():
    # 'phở có' must NOT match landmark 'Phố Cổ' (ở ≠ ố), nor 'sáng' the key 'sang'.
    assert not compat_token_seq("quán phở có chỗ ngồi ngoài trời", "Phố Cổ")
    assert not compat_token_seq("quán ăn sáng", "sang")


def test_compat_token_seq_unaccented_and_exact_match():
    # unaccented input still matches (folding's whole purpose); exact accented too.
    assert compat_token_seq("cafe gan pho co", "Phố Cổ")
    assert compat_token_seq("cafe gần phố cổ", "Phố Cổ")
    assert compat_token_seq("nhà hàng sang trọng", "sang")


def test_compat_token_seq_token_boundary():
    # 'park' must not fire inside 'parking'; 'late' not inside 'chocolate';
    # 'ho tay' not inside 'cho tay'.
    assert not compat_token_seq("bãi parking rộng", "park")
    assert not compat_token_seq("bánh chocolate", "late")
    assert not compat_token_seq("tìm chỗ tây ba lô", "Hồ Tây")
    assert compat_token_seq("công viên lê văn tám", "công viên")


def test_token_key_matches_permissive_on_expansion():
    # key arriving only via abbreviation expansion (not literally in raw) still matches;
    # a conflicting diacritic literally in raw is rejected.
    assert token_key_matches("quan ca phe o quan 1", "quan cf o q1", "Quận 1")
    assert not token_key_matches("quan an sang", "quán ăn sáng", "sang")
    assert token_key_matches("nha hang sang trong", "nhà hàng sang trọng", "sang")


# --- Fix 2: diacritic-aware Vietnamese common-word flagging ---

def test_query_common_flags_superlative_particle():
    assert "nhat" in query_common_tokens("địa điểm nổi tiếng nhất")
    assert "nhat" in query_common_tokens("cay xang gan nhat")  # unaccented too


def test_query_common_does_not_swallow_food_subject_cha():
    # 'chả' (food) folds to 'cha' but the common word is plain 'cha' -> not flagged.
    assert "cha" not in query_common_tokens("quán bún chả")
