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
from .normalize import STOPWORDS, contains_token_seq, expand_query, fold

# A leftover query term is a "distinctive subject" (drives the subject hard-filter)
# only if it is a rare *word that names POIs* — a dish/brand like "bún chả" (df=1),
# not a common descriptor ("mua sắm" df=13), a time token ("24h"), or a short generic
# verb ("ăn"). Hence: alphabetic, length >= 3, and appears in <= 2 POI names. SPEC §6.
DISTINCTIVE_DF_MAX = 2
DISTINCTIVE_MIN_LEN = 3

# Category keywords: folded query term -> canonical dataset category. Longest match wins.
CATEGORY_KEYWORDS: dict[str, str] = {
    "quan ca phe": "Quán cà phê", "ca phe": "Quán cà phê", "coffee": "Quán cà phê",
    "nha hang": "Nhà hàng", "quan an": "Nhà hàng", "restaurant": "Nhà hàng",
    "khach san": "Khách sạn", "hotel": "Khách sạn", "resort": "Khách sạn",
    "trung tam thuong mai": "Trung tâm thương mại", "mall": "Trung tâm thương mại",
    "atm": "ATM", "rut tien": "ATM",
    "tram xang": "Trạm xăng", "cay xang": "Trạm xăng", "xang dau": "Trạm xăng",
    "cong vien": "Công viên", "park": "Công viên",
    "benh vien": "Bệnh viện", "hospital": "Bệnh viện",
    "nha thuoc": "Nhà thuốc", "hieu thuoc": "Nhà thuốc", "pharmacy": "Nhà thuốc",
    "rap phim": "Rạp phim", "rap chieu phim": "Rạp phim", "cinema": "Rạp phim",
    "tram sac dien": "Trạm sạc điện", "tram sac": "Trạm sạc điện", "sac dien": "Trạm sạc điện",
    "diem tham quan": "Điểm tham quan", "tham quan": "Điểm tham quan", "du lich": "Điểm tham quan",
}

# Attribute canonicalizer: folded query phrase -> canonical taxonomy attribute (the fixed 10).
ATTRIBUTE_KEYWORDS: dict[str, str] = {
    "yen tinh": "yên tĩnh", "tinh lang": "yên tĩnh", "quiet": "yên tĩnh",
    "wifi": "wifi",
    "lam viec": "phù hợp làm việc", "phu hop lam viec": "phù hợp làm việc", "work": "phù hợp làm việc",
    "gia dinh": "phù hợp gia đình", "phu hop gia dinh": "phù hợp gia đình", "tre em": "phù hợp gia đình",
    "lang man": "lãng mạn", "hen ho": "lãng mạn", "romantic": "lãng mạn",
    "mo khuya": "mở khuya", "khuya": "mở khuya", "late": "mở khuya",
    "gan bien": "gần biển", "view bien": "gần biển", "beach": "gần biển",
    "bai do xe": "bãi đỗ xe", "do xe": "bãi đỗ xe", "parking": "bãi đỗ xe",
    "check in": "check-in", "song ao": "check-in", "view dep": "check-in",
    "24/7": "24/7",
}


class Parser:
    def __init__(self, pois: Sequence[POI], gazetteer: Gazetteer):
        self.gazetteer = gazetteer
        self.categories = {p.category for p in pois}
        self.city_vocab = {fold(p.city): p.city for p in pois}
        self._cat_keys = sorted(CATEGORY_KEYWORDS, key=len, reverse=True)   # longest first
        self._attr_keys = sorted(ATTRIBUTE_KEYWORDS, key=len, reverse=True)
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
            if key in expanded or key in folded:
                cand = CATEGORY_KEYWORDS[key]
                if cand in self.categories:
                    category = cand
                    consumed.update(key.split())
                    break

        required: list[str] = []
        for key in self._attr_keys:
            if f"{key}" in hay:
                consumed.update(key.split())
                canon = ATTRIBUTE_KEYWORDS[key]
                if canon not in required:
                    required.append(canon)

        city = next((canon for key, canon in self.city_vocab.items() if key in hay), None)
        if city:
            consumed.update(fold(city).split())
        open_after = "22:00" if "mo khuya" in hay or "khuya" in hay else None
        # Resolve against the expanded haystack so abbreviations resolve too:
        # "q1" -> "quan 1" now matches the gazetteer's district centroid (FR-2).
        anchor = self.gazetteer.resolve(hay)
        if anchor:
            consumed.update(fold(anchor.name).split())
        # Lift any district reference into the structured field, token-boundary matched
        # (shortest key first -> "Quận 1"); "quan 1" must NOT fire inside "quan 10".
        district = next(
            (disp for key, (_, _, disp) in self.gazetteer.districts.items()
             if contains_token_seq(hay, key)),
            None,
        )
        if district:
            consumed.update(fold(district).split())

        # Residual = query content the parse did NOT explain (after stopwords). Its
        # presence blocks the category hard-filter (guards mis-parses P019/P055);
        # its distinctive (rare) tokens become the subject hard-filter.
        residual = [
            t for t in dict.fromkeys(exp_tokens)
            if t not in consumed and t not in STOPWORDS and len(t) >= 2 and not t.isdigit()
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
            city=city,
            district=district,
            content_terms=content_terms,
            has_residual=bool(residual),
        )
