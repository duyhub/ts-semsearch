"""Rule-based query intent extraction (SPEC §7; PRD FR-2).

Closed to the dataset's own vocabularies — the 12 category names and the
10-attribute taxonomy — via curated keyword maps (category keywords + the
attribute canonicalizer). Public-data enrichment (synonyms, English, non-
accented forms) is allowed and not fitted to eval queries (NFR-6). The LLM
parser (FR-4) layers on top later; rules run unconditionally.
"""
from __future__ import annotations

from collections import Counter
from typing import Sequence

from .data import POI, QueryIntent, content_tokens
from .geo import Gazetteer
from .normalize import (
    STOPWORDS,
    expand_query,
    fold,
    query_common_tokens,
    token_key_matches,
)

# A leftover query term is a "distinctive subject" (drives the subject hard-filter)
# only if it is a rare *word that names POIs* — a dish/brand like "bún chả" (df=1),
# not a common descriptor ("mua sắm" df=13), a time token ("24h"), or a short generic
# verb ("ăn"). Hence: alphabetic, length >= 3, and appears in <= 2 POI names. SPEC §6.
DISTINCTIVE_DF_MAX = 2
DISTINCTIVE_MIN_LEN = 3

# Category keywords: DISPLAY (diacritic) query phrase -> canonical dataset category.
# Keys carry correct diacritics so matching is both token-boundary and diacritic-
# compatible (Fix 1: 'park' no longer fires inside 'parking'). Longest match wins.
# Drink compounds ('trà sữa'/'trà đá'/'trà chanh') route to the café category (Fix 3a);
# bare 'trà' is deliberately excluded — it would misfire on the district 'Sơn Trà'.
CATEGORY_KEYWORDS: dict[str, str] = {
    "quán cà phê": "Quán cà phê", "cà phê": "Quán cà phê", "coffee": "Quán cà phê",
    "trà sữa": "Quán cà phê", "trà đá": "Quán cà phê", "trà chanh": "Quán cà phê",
    "nhà hàng": "Nhà hàng", "quán ăn": "Nhà hàng", "restaurant": "Nhà hàng",
    "khách sạn": "Khách sạn", "hotel": "Khách sạn", "resort": "Khách sạn",
    "trung tâm thương mại": "Trung tâm thương mại", "mall": "Trung tâm thương mại",
    "atm": "ATM", "rút tiền": "ATM",
    "trạm xăng": "Trạm xăng", "cây xăng": "Trạm xăng", "xăng dầu": "Trạm xăng",
    "công viên": "Công viên", "park": "Công viên",
    "bệnh viện": "Bệnh viện", "hospital": "Bệnh viện",
    "nhà thuốc": "Nhà thuốc", "hiệu thuốc": "Nhà thuốc", "pharmacy": "Nhà thuốc",
    "rạp phim": "Rạp phim", "rạp chiếu phim": "Rạp phim", "cinema": "Rạp phim",
    "trạm sạc điện": "Trạm sạc điện", "trạm sạc": "Trạm sạc điện", "sạc điện": "Trạm sạc điện",
    "điểm tham quan": "Điểm tham quan", "tham quan": "Điểm tham quan", "du lịch": "Điểm tham quan",
}

# Price-direction keywords: DISPLAY (diacritic) phrase -> "cheap" | "expensive". Read
# from the haystack independently of the stopword/residual logic, so "rẻ"/"sang" inform
# price without ever becoming spurious subjects. Keys carry correct diacritics so token-
# boundary + diacritic-compatible matching (Fix 1) lets luxury "sang" match while the
# morning word "sáng" does NOT. Bare "đắt" is DELIBERATELY excluded — folded "đặt"
# ("to book", e.g. "đặt bàn") collides with "đắt" ("expensive"). On a cheap+expensive
# conflict, cheap wins (the far more common intent).
PRICE_KEYWORDS: dict[str, str] = {
    "giá rẻ": "cheap", "bình dân": "cheap", "tiết kiệm": "cheap",
    "rẻ": "cheap", "cheap": "cheap", "budget": "cheap",
    "sang trọng": "expensive", "sang chảnh": "expensive", "cao cấp": "expensive",
    "sang": "expensive", "luxury": "expensive",
}

# Attribute canonicalizer: DISPLAY (diacritic) query phrase -> canonical taxonomy
# attribute (the fixed 10). Keys carry correct diacritics so matching is token-boundary
# + diacritic-compatible (Fix 1): 'late' no longer fires inside 'chocolate', and 'đỗ xe'
# is not tripped by 'độ xe' (modify a vehicle).
ATTRIBUTE_KEYWORDS: dict[str, str] = {
    "yên tĩnh": "yên tĩnh", "tĩnh lặng": "yên tĩnh", "quiet": "yên tĩnh",
    "wifi": "wifi",
    "làm việc": "phù hợp làm việc", "phù hợp làm việc": "phù hợp làm việc", "work": "phù hợp làm việc",
    "gia đình": "phù hợp gia đình", "phù hợp gia đình": "phù hợp gia đình", "trẻ em": "phù hợp gia đình",
    "lãng mạn": "lãng mạn", "hẹn hò": "lãng mạn", "romantic": "lãng mạn",
    "mở khuya": "mở khuya", "khuya": "mở khuya", "late": "mở khuya",
    "gần biển": "gần biển", "view biển": "gần biển", "beach": "gần biển",
    "bãi đỗ xe": "bãi đỗ xe", "đỗ xe": "bãi đỗ xe", "parking": "bãi đỗ xe",
    "check in": "check-in", "sống ảo": "check-in", "view đẹp": "check-in",
    "24/7": "24/7",
}


class Parser:
    def __init__(self, pois: Sequence[POI], gazetteer: Gazetteer):
        self.gazetteer = gazetteer
        self.categories = {p.category for p in pois}
        self.city_vocab = {fold(p.city): p.city for p in pois}
        # longest folded match first (keys are diacritic display forms)
        self._cat_keys = sorted(CATEGORY_KEYWORDS, key=lambda k: len(fold(k)), reverse=True)
        self._attr_keys = sorted(ATTRIBUTE_KEYWORDS, key=lambda k: len(fold(k)), reverse=True)
        # corpus document-frequency of subject tokens, for distinctive-term detection
        self._df: Counter[str] = Counter()
        for p in pois:
            self._df.update(content_tokens(p))

    def parse(self, text: str) -> QueryIntent:
        folded = fold(text)
        exp_tokens = expand_query(text)
        expanded = " ".join(exp_tokens)
        hay = f" {expanded} {folded} "
        consumed: set[str] = set()  # tokens explained by a recognized vocab element

        category = None
        for key in self._cat_keys:
            # token-boundary + diacritic-compatible (Fix 1): 'park' won't fire in 'parking'
            if token_key_matches(hay, text, key):
                cand = CATEGORY_KEYWORDS[key]
                if cand in self.categories:
                    category = cand
                    consumed.update(fold(key).split())
                    break

        required: list[str] = []
        for key in self._attr_keys:
            if token_key_matches(hay, text, key):
                consumed.update(fold(key).split())
                canon = ATTRIBUTE_KEYWORDS[key]
                if canon not in required:
                    required.append(canon)

        city = next(
            (disp for _fkey, disp in self.city_vocab.items()
             if token_key_matches(hay, text, disp)),
            None,
        )
        if city:
            consumed.update(fold(city).split())
        # "mở khuya" implies an open-after-22:00 preference (derived from the matched attr).
        open_after = "22:00" if "mở khuya" in required else None
        # Resolve against the expanded haystack (so "q1" -> "quan 1" resolves, FR-2) plus
        # the raw query for diacritic compatibility (Fix 1: 'phở có' won't anchor to Phố Cổ).
        anchor = self.gazetteer.resolve(hay, text)
        if anchor:
            consumed.update(fold(anchor.name).split())
        # Lift any district reference into the structured field, token-boundary + diacritic-
        # compatible ("quan 1" must NOT fire inside "quan 10"; expansion of "q1" still works).
        district = next(
            (disp for _fkey, (_, _, disp) in self.gazetteer.districts.items()
             if token_key_matches(hay, text, disp)),
            None,
        )
        if district:
            consumed.update(fold(district).split())

        # Price direction (affordability intent). Token-boundary + diacritic-compatible on
        # the haystack (Fix 1: 'sáng' won't fire luxury 'sang'); matched price tokens are
        # consumed so "bình dân"/"giá rẻ" never leak into residual.
        price_dirs = set()
        for key, direction in PRICE_KEYWORDS.items():
            if token_key_matches(hay, text, key):
                price_dirs.add(direction)
                consumed.update(fold(key).split())
        price_pref = "cheap" if "cheap" in price_dirs else ("expensive" if price_dirs else None)

        # Residual = query content the parse did NOT explain (after stopwords and the
        # Vietnamese common-word list, Fix 2). Its presence blocks the category hard-filter
        # (guards mis-parses P019/P055); its distinctive (rare) tokens become the subject
        # hard-filter. Diacritic-aware common-word dropping keeps 'chả' (food) a subject
        # while removing the superlative particle 'nhất'.
        common = query_common_tokens(text)
        residual = [
            t for t in dict.fromkeys(exp_tokens)
            if t not in consumed and t not in STOPWORDS and t not in common
            and len(t) >= 2 and not t.isdigit()
        ]
        content_terms = [
            t for t in residual
            if t.isalpha() and len(t) >= DISTINCTIVE_MIN_LEN
            and 1 <= self._df.get(t, 0) <= DISTINCTIVE_DF_MAX
        ]

        return QueryIntent(
            raw=text,
            normalized=expanded,
            category=category,
            anchor=anchor,
            required_attrs=required,
            soft_prefs=[],
            open_after=open_after,
            price_pref=price_pref,
            city=city,
            district=district,
            content_terms=content_terms,
            has_residual=bool(residual),
            residual_terms=residual,
        )
