#!/usr/bin/env python3
"""Deterministic LABELED Vietnamese eval-query generator over the 1000-POI stress corpus (Worker E).

Produces ~150 evaluation queries (ids ``SQ001``..``SQ150``) against
``data/synth/synth_dataset.xlsx`` (the official 111 POIs verbatim + 889 ``SYN####``
distractors). This is a stress-test eval set that measures retrieval robustness at
1000 POIs — a complement to the official 60-query public set.

Ground truth BY CONSTRUCTION — the single most important property:

    Labels are NEVER produced by running the search engine under test. They are
    computed from EXPLICIT predicates over raw POI fields (category, city, district,
    attributes, opening_hours, price_level, name). Nothing in this file imports
    ``pipeline`` / ``retrieve`` / ``rank`` / ``engines`` — that would be circular and
    worthless. The only ``semsearch`` imports are the data loaders and the pure
    text-normalization primitive ``fold`` (used inside string predicates).

Construction is TARGET-FIRST: sample a target POI, build a Vietnamese query from its
ACTUAL fields, define the query's constraint predicate explicitly, then set
``expected_ids`` = every POI in the corpus satisfying that predicate, ordered by a
defensible quality key. If the predicate is too generic (>8 matches) it is tightened
(add an attribute / narrow to district) until 1-6 remain; the target is always a
candidate by construction, so the set is never empty.

Determinism: driven ONLY by ``random.Random(seed)`` — no global ``random``, no
``time``, no network. Same seed -> identical JSON content.

    uv run python scripts/gen_synth_queries.py
    uv run python scripts/gen_synth_queries.py --n 150 --seed 20260711
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from semsearch.data import POI, load_eval, load_pois  # noqa: E402
from semsearch.normalize import contains_token_seq, fold  # noqa: E402

# --------------------------------------------------------------------------- #
# Vocabulary — reuses the official eval set's query_category / difficulty /    #
# skills_tested values so downstream reporting groups cleanly.                 #
# --------------------------------------------------------------------------- #

# Category (canonical corpus string) -> natural Vietnamese query nouns (variants for variety).
CAT_QUERY: dict[str, list[str]] = {
    "Quán cà phê": ["quán cà phê", "quán cafe", "cà phê"],
    "Nhà hàng": ["nhà hàng", "quán ăn"],
    "Khách sạn": ["khách sạn"],
    "Trung tâm thương mại": ["trung tâm thương mại", "trung tâm mua sắm"],
    "ATM": ["ATM", "cây ATM", "trụ ATM"],
    "Trạm xăng": ["trạm xăng", "cây xăng"],
    "Điểm tham quan": ["điểm tham quan", "địa điểm tham quan", "chỗ tham quan"],
    "Bệnh viện": ["bệnh viện"],
    "Rạp phim": ["rạp phim", "rạp chiếu phim"],
    "Công viên": ["công viên"],
    "Trạm sạc điện": ["trạm sạc điện", "trạm sạc xe điện"],
    "Nhà thuốc": ["nhà thuốc", "hiệu thuốc"],
}

# Category -> English noun (for the mixed EN-VN degradation).
CAT_EN: dict[str, list[str]] = {
    "Quán cà phê": ["cafe", "coffee shop"],
    "Nhà hàng": ["restaurant"],
    "Khách sạn": ["hotel"],
    "Trung tâm thương mại": ["mall", "shopping mall"],
    "ATM": ["ATM"],
    "Trạm xăng": ["gas station"],
    "Điểm tham quan": ["attraction"],
    "Bệnh viện": ["hospital"],
    "Rạp phim": ["cinema"],
    "Công viên": ["park"],
    "Trạm sạc điện": ["EV charging station"],
    "Nhà thuốc": ["pharmacy"],
}

# Closed 10-attribute taxonomy -> Vietnamese query phrases. Attributes are matched by
# EXACT membership in POI.attributes (canonical strings), so predicates stay precise.
ATTR_QUERY: dict[str, list[str]] = {
    "yên tĩnh": ["yên tĩnh"],
    "wifi": ["có wifi", "wifi"],
    "phù hợp làm việc": ["để làm việc", "ngồi làm việc"],
    "phù hợp gia đình": ["cho gia đình", "phù hợp gia đình"],
    "lãng mạn": ["lãng mạn", "để hẹn hò"],
    "mở khuya": ["mở khuya"],
    "gần biển": ["gần biển"],
    "bãi đỗ xe": ["có chỗ đậu xe", "có bãi đỗ xe"],
    "check-in": ["để check-in", "sống ảo đẹp"],
    "24/7": ["24/7"],
}

# Attribute -> English rendering, split by natural placement around the noun.
EN_PREFIX_ATTR: dict[str, str] = {
    "yên tĩnh": "quiet",
    "lãng mạn": "romantic",
    "phù hợp gia đình": "family friendly",
}
EN_SUFFIX_ATTR: dict[str, str] = {
    "wifi": "with wifi",
    "phù hợp làm việc": "to work",
    "bãi đỗ xe": "with parking",
    "gần biển": "near the beach",
    "check-in": "for photos",
    "mở khuya": "open late",
    "24/7": "24/7",
}

# City (canonical) -> display variants and abbreviation (for the abbrev degradation).
CITY_DISP: dict[str, list[str]] = {
    "TP.HCM": ["TP.HCM", "Sài Gòn"],
    "Hà Nội": ["Hà Nội"],
    "Đà Nẵng": ["Đà Nẵng"],
    "Đà Lạt": ["Đà Lạt"],
}
CITY_ABBR: dict[str, str] = {"TP.HCM": "tphcm", "Hà Nội": "hn", "Đà Nẵng": "dn", "Đà Lạt": "dl"}

# Superlative phrasing by category (order by rating desc). Falls back to DEFAULT_SUPERLATIVE.
SUPERLATIVE_WORDS: dict[str, list[str]] = {
    "Quán cà phê": ["ngon nhất", "tốt nhất", "xịn nhất"],
    "Nhà hàng": ["ngon nhất", "tốt nhất"],
    "Khách sạn": ["tốt nhất", "sang nhất"],
    "Điểm tham quan": ["đẹp nhất", "nổi tiếng nhất"],
    "Công viên": ["đẹp nhất", "rộng nhất"],
    "Trung tâm thương mại": ["lớn nhất", "nổi tiếng nhất"],
    "Rạp phim": ["tốt nhất", "hiện đại nhất"],
}
DEFAULT_SUPERLATIVE = ["tốt nhất", "nổi tiếng nhất"]
SUPERLATIVE_CATS = frozenset(SUPERLATIVE_WORDS)

# Categories where a price intent (cheap / upscale) reads naturally.
PRICE_CATS = frozenset({"Quán cà phê", "Nhà hàng", "Khách sạn"})
PRICE_CHEAP = ["giá rẻ", "giá bình dân"]
PRICE_UPSCALE = ["sang trọng", "cao cấp"]

# Open-late / 24-7 phrasing, and the categories where each reads naturally.
LATE_CATS = frozenset({"Quán cà phê", "Nhà hàng", "Nhà thuốc", "Rạp phim"})
H247_CATS = frozenset({"ATM", "Trạm xăng", "Bệnh viện", "Nhà thuốc"})
LATE_WORDS = ["mở khuya", "mở cửa khuya"]
H247_WORDS = ["24/7", "mở cửa 24/24"]

# Need-based paraphrases (NO category word in the text) -> the category (+ hidden attrs)
# they resolve to. Every one is a Hard query.
NEEDS: list[dict] = [
    {"phrases": ["chỗ ngồi làm việc cả buổi", "nơi vừa nhâm nhi vừa làm việc lâu"],
     "category": "Quán cà phê", "attrs": ["phù hợp làm việc"]},
    {"phrases": ["chỗ ngồi học bài yên tĩnh cả ngày", "nơi ngồi ôn thi lâu"],
     "category": "Quán cà phê", "attrs": ["yên tĩnh"]},
    {"phrases": ["chỗ rút tiền mặt gấp", "nơi rút tiền tiện lợi"],
     "category": "ATM", "attrs": []},
    {"phrases": ["chỗ đổ xăng cho xe", "nơi bơm xăng tiện đường"],
     "category": "Trạm xăng", "attrs": []},
    {"phrases": ["nơi khám bệnh khi ốm", "chỗ cấp cứu ban đêm"],
     "category": "Bệnh viện", "attrs": []},
    {"phrases": ["chỗ coi phim cuối tuần", "nơi xem phim bom tấn"],
     "category": "Rạp phim", "attrs": []},
    {"phrases": ["nơi mua sắm quần áo cuối tuần", "chỗ dạo mua đồ cả ngày"],
     "category": "Trung tâm thương mại", "attrs": []},
    {"phrases": ["chỗ sạc pin ô tô điện", "nơi cắm sạc xe điện"],
     "category": "Trạm sạc điện", "attrs": []},
    {"phrases": ["chỗ mua thuốc gấp buổi tối", "nơi mua thuốc ho"],
     "category": "Nhà thuốc", "attrs": []},
    {"phrases": ["chỗ nghỉ chân qua đêm", "nơi ngủ lại một đêm"],
     "category": "Khách sạn", "attrs": []},
    {"phrases": ["nơi hẹn hò lãng mạn buổi tối", "chỗ đưa người yêu đi ăn tối"],
     "category": "Nhà hàng", "attrs": ["lãng mạn"]},
    {"phrases": ["chỗ cho cả nhà quây quần ăn uống", "nơi ăn tối đông người có trẻ nhỏ"],
     "category": "Nhà hàng", "attrs": ["phù hợp gia đình"]},
    {"phrases": ["chỗ đưa trẻ con ra chạy nhảy", "nơi đi dạo hóng mát cuối tuần"],
     "category": "Công viên", "attrs": []},
]

# Categories whose district+category signal alone is strong enough to read as Easy.
RARE_CATS = frozenset(
    {"Bệnh viện", "Nhà thuốc", "Trạm sạc điện", "Công viên", "Rạp phim", "Điểm tham quan"}
)

# Per-family metadata for the row's informational fields (all values reuse the official vocab).
FAMILY_QCAT: dict[str, str] = {
    "attribute-seek": "Attribute Search",
    "category-location": "Location-Aware Search",
    "paraphrase": "Intent-Based Search",
    "superlative": "Discovery Search",
    "price": "Semantic Search",
    "open-late": "Attribute Search",
    "brand": "POI Search",
}
FAMILY_INTENT: dict[str, str] = {
    "attribute-seek": "Category Search",
    "category-location": "Category Search",
    "paraphrase": "Category Search",
    "superlative": "Discovery Search",
    "price": "Category Search",
    "open-late": "Category Search",
    "brand": "POI Search",
}
FAMILY_SKILLS: dict[str, list[str]] = {
    "attribute-seek": ["Attribute", "Semantic", "Location"],
    "category-location": ["Category", "Location"],
    "paraphrase": ["Intent", "Semantic", "Location"],
    "superlative": ["Ranking", "Semantic", "Location"],
    "price": ["Price", "Semantic", "Location"],
    "open-late": ["Time", "Attribute", "Location"],
    "brand": ["POI", "Location"],
}
FAMILY_SIGNALS: dict[str, list[str]] = {
    "attribute-seek": ["attributes", "relevance", "rating"],
    "category-location": ["location", "relevance"],
    "paraphrase": ["semantic", "attributes", "location"],
    "superlative": ["rating", "relevance", "popularity"],
    "price": ["price", "relevance", "rating"],
    "open-late": ["opening_hours", "relevance"],
    "brand": ["relevance", "location"],
}

# Relative sampling weights (need not sum to 1 — weighted_choice normalizes by total).
FAMILY_W: dict[str, float] = {
    "attribute-seek": 0.30,
    "category-location": 0.23,
    "brand": 0.15,
    "paraphrase": 0.08,
    "superlative": 0.07,
    "price": 0.06,
    "open-late": 0.06,
}

# Degradation styles applied to the TEXT ONLY (labels unchanged). Relative weights.
STYLE_W: dict[str, float] = {
    "none": 0.45, "diacritics": 0.22, "typo": 0.12, "both": 0.06, "abbrev": 0.09, "mixed": 0.06,
}
# skills_tested flag recorded for each degradation (official 'MixedLanguage' where it fits).
STYLE_FLAGS: dict[str, list[str]] = {
    "none": [], "diacritics": ["Diacritics"], "typo": ["Typo"],
    "both": ["Diacritics", "Typo"], "abbrev": ["Abbreviation"], "mixed": ["MixedLanguage"],
}

CAP = 6        # tighten toward this many candidates
HARD_CAP = 8   # never label more than this — resample if still exceeded

_QUAN_RE = re.compile(r"Quận\s+(\d+)$")
_HOURS_RE = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


# --------------------------------------------------------------------------- #
# Hours parsing — an explicit little parser (never the engine under test).     #
# --------------------------------------------------------------------------- #
def is_late(hours: str | None) -> bool:
    """True iff the POI is open late: 24/7, overnight (closes past midnight), or closes >= 23:00."""
    if not hours:
        return False
    h = hours.strip()
    if h == "24/7":
        return True
    m = _HOURS_RE.match(h)
    if not m:
        return False
    open_min = int(m.group(1)) * 60 + int(m.group(2))
    close_min = int(m.group(3)) * 60 + int(m.group(4))
    if close_min < open_min:  # crosses midnight, e.g. 18:00-03:00
        return True
    return close_min >= 23 * 60


def is_247(hours: str | None) -> bool:
    return bool(hours) and hours.strip() == "24/7"


# --------------------------------------------------------------------------- #
# Deterministic helpers                                                        #
# --------------------------------------------------------------------------- #
def weighted_choice(rng: Random, weights: dict) -> str:
    keys = list(weights)
    total = sum(weights.values())
    r = rng.random() * total
    upto = 0.0
    for k in keys:
        upto += weights[k]
        if r <= upto:
            return k
    return keys[-1]


def sample_where(rng: Random, pois: list[POI], pred: Callable[[POI], bool], tries: int = 60) -> POI | None:
    for _ in range(tries):
        p = pois[rng.randrange(len(pois))]
        if pred(p):
            return p
    return None


def strip_diacritics(s: str) -> str:
    """Remove Vietnamese diacritics, preserving case/spacing/'/' (đ->d). Text-degradation only."""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", s)


def inject_typo(rng: Random, text: str) -> tuple[str, bool]:
    """Swap two adjacent chars in one word of length >= 4. Returns (text, applied?)."""
    words = text.split(" ")
    idxs = [i for i, w in enumerate(words) if len(w) >= 4]
    if not idxs:
        return text, False
    for i in rng.sample(idxs, len(idxs)):
        w = words[i]
        for p in rng.sample(range(len(w) - 1), len(w) - 1):
            if w[p] != w[p + 1]:
                words[i] = w[:p] + w[p + 1] + w[p] + w[p + 2:]
                return " ".join(words), True
    return text, False


def name_contains(poi: POI, fragment: str) -> bool:
    """True iff the folded fragment appears as a token subsequence of the POI's folded name+brand."""
    hay = fold(f"{poi.name} {poi.brand or ''}")
    return contains_token_seq(hay, fold(fragment))


# --------------------------------------------------------------------------- #
# Query spec + predicate                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    family: str
    category: str | None            # None only for brand family
    city: str | None
    district: str | None
    attrs: list[str]                # ALL required attributes (predicate)
    hours_mode: str | None          # None | 'late' | '247'
    name_fragment: str | None       # brand family
    order: str                      # 'quality' | 'rating' | 'price_asc' | 'price_desc'
    hidden_attrs: set[str] = field(default_factory=set)  # attrs baked into a need phrase (not re-rendered)
    need_phrases: list[str] = field(default_factory=list)  # paraphrase family
    price_dir: str | None = None    # 'cheap' | 'upscale'


def predicate_for(spec: Spec) -> Callable[[POI], bool]:
    cat = spec.category
    city = spec.city
    district = spec.district
    attrs = tuple(spec.attrs)
    hours_mode = spec.hours_mode
    fragment = spec.name_fragment

    def pred(poi: POI) -> bool:
        if cat is not None and poi.category != cat:
            return False
        if city is not None and poi.city != city:
            return False
        if district is not None and poi.district != district:
            return False
        for a in attrs:
            if a not in poi.attributes:
                return False
        if hours_mode == "late" and not is_late(poi.opening_hours):
            return False
        if hours_mode == "247" and not is_247(poi.opening_hours):
            return False
        if fragment is not None and not name_contains(poi, fragment):
            return False
        return True

    return pred


def order_key(order: str) -> Callable[[POI], tuple]:
    if order == "rating":
        return lambda p: (-p.rating, -p.review_count, p.poi_id)
    if order == "price_asc":
        return lambda p: (p.price_level if p.price_level is not None else 99, -p.rating, p.poi_id)
    if order == "price_desc":
        return lambda p: (-(p.price_level if p.price_level is not None else 0), -p.rating, p.poi_id)
    return lambda p: (-p.rating, -p.review_count, -p.popularity, p.poi_id)


def candidates_for(spec: Spec, pois: list[POI]) -> list[POI]:
    pred = predicate_for(spec)
    return [p for p in pois if pred(p)]


def select_candidates(spec: Spec, target: POI, pois: list[POI]) -> list[POI] | None:
    """Tighten the predicate until 1..HARD_CAP candidates remain. The target always
    satisfies the predicate by construction, so the set is never empty. Returns None if
    even after adding every usable target attribute the set still exceeds HARD_CAP."""
    cands = candidates_for(spec, pois)
    # First narrow city -> district if too generic.
    if len(cands) > CAP and spec.district is None and target.district:
        spec.city, spec.district = target.city, target.district
        cands = candidates_for(spec, pois)
    # Then add the target's own attributes (only ones with a query phrase) one at a time.
    usable = [a for a in target.attributes if a in ATTR_QUERY and a not in spec.attrs]
    i = 0
    while len(cands) > CAP and i < len(usable):
        spec.attrs.append(usable[i])
        i += 1
        cands = candidates_for(spec, pois)
    if len(cands) > HARD_CAP:
        return None
    return cands


# --------------------------------------------------------------------------- #
# Family builders — each returns (spec, candidates) or None (resample).        #
# --------------------------------------------------------------------------- #
def _has_query_attr(poi: POI) -> bool:
    return any(a in ATTR_QUERY for a in poi.attributes)


def build_attribute_seek(rng: Random, pois: list[POI]) -> tuple[Spec, list[POI]] | None:
    target = sample_where(rng, pois, _has_query_attr)
    if target is None:
        return None
    a0 = rng.choice([a for a in target.attributes if a in ATTR_QUERY])
    spec = Spec(family="attribute-seek", category=target.category, city=target.city,
                district=target.district, attrs=[a0], hours_mode=None, name_fragment=None,
                order="quality")
    cands = select_candidates(spec, target, pois)
    return None if cands is None else (spec, cands)


def build_category_location(rng: Random, pois: list[POI]) -> tuple[Spec, list[POI]] | None:
    target = pois[rng.randrange(len(pois))]
    scope = "district" if rng.random() < 0.6 else "city"
    spec = Spec(family="category-location", category=target.category, city=target.city,
                district=(target.district if scope == "district" else None), attrs=[],
                hours_mode=None, name_fragment=None, order="quality")
    cands = select_candidates(spec, target, pois)
    return None if cands is None else (spec, cands)


def build_paraphrase(rng: Random, pois: list[POI]) -> tuple[Spec, list[POI]] | None:
    need = rng.choice(NEEDS)
    req = set(need["attrs"])
    target = sample_where(
        rng, pois,
        lambda p: p.category == need["category"] and req.issubset(set(p.attributes)),
    )
    if target is None:
        return None
    spec = Spec(family="paraphrase", category=need["category"], city=target.city,
                district=target.district, attrs=list(need["attrs"]), hours_mode=None,
                name_fragment=None, order="quality", hidden_attrs=set(need["attrs"]),
                need_phrases=list(need["phrases"]))
    cands = select_candidates(spec, target, pois)
    return None if cands is None else (spec, cands)


def build_superlative(rng: Random, pois: list[POI]) -> tuple[Spec, list[POI]] | None:
    target = sample_where(rng, pois, lambda p: p.category in SUPERLATIVE_CATS)
    if target is None:
        return None
    spec = Spec(family="superlative", category=target.category, city=target.city,
                district=target.district, attrs=[], hours_mode=None, name_fragment=None,
                order="rating")
    cands = select_candidates(spec, target, pois)
    return None if cands is None else (spec, cands)


def build_price(rng: Random, pois: list[POI]) -> tuple[Spec, list[POI]] | None:
    target = sample_where(rng, pois, lambda p: p.category in PRICE_CATS and p.price_level is not None)
    if target is None:
        return None
    direction = "cheap" if rng.random() < 0.5 else "upscale"
    spec = Spec(family="price", category=target.category, city=target.city,
                district=target.district, attrs=[], hours_mode=None, name_fragment=None,
                order=("price_asc" if direction == "cheap" else "price_desc"),
                price_dir=direction)
    cands = select_candidates(spec, target, pois)
    return None if cands is None else (spec, cands)


def build_open_late(rng: Random, pois: list[POI]) -> tuple[Spec, list[POI]] | None:
    def ok(p: POI) -> bool:
        if is_247(p.opening_hours) and p.category in H247_CATS:
            return True
        return is_late(p.opening_hours) and p.category in LATE_CATS

    target = sample_where(rng, pois, ok)
    if target is None:
        return None
    mode = "247" if is_247(target.opening_hours) else "late"
    spec = Spec(family="open-late", category=target.category, city=target.city,
                district=target.district, attrs=[], hours_mode=mode, name_fragment=None,
                order="quality")
    cands = select_candidates(spec, target, pois)
    return None if cands is None else (spec, cands)


def build_brand(rng: Random, pois: list[POI]) -> tuple[Spec, list[POI]] | None:
    target = sample_where(rng, pois, lambda p: len(p.name) >= 3)
    if target is None:
        return None
    toks = target.name.split()
    options: list[str] = []
    if target.brand and len(target.brand) >= 3:
        options.append(target.brand)
    for length in (2, 3):
        if len(toks) >= length:
            options.append(" ".join(toks[:length]))
    options.append(target.name)
    seen: set[str] = set()
    for frag in options:
        if frag in seen:
            continue
        seen.add(frag)
        spec = Spec(family="brand", category=None, city=target.city, district=target.district,
                    attrs=[], hours_mode=None, name_fragment=frag, order="quality")
        cands = candidates_for(spec, pois)
        if 1 <= len(cands) <= HARD_CAP:
            return spec, cands
    return None


BUILDERS: dict[str, Callable[[Random, list[POI]], "tuple[Spec, list[POI]] | None"]] = {
    "attribute-seek": build_attribute_seek,
    "category-location": build_category_location,
    "paraphrase": build_paraphrase,
    "superlative": build_superlative,
    "price": build_price,
    "open-late": build_open_late,
    "brand": build_brand,
}


# --------------------------------------------------------------------------- #
# Rendering: spec + degradation style -> Vietnamese query text                 #
# --------------------------------------------------------------------------- #
def _render_attrs(spec: Spec) -> list[str]:
    """Attributes to render as phrases (all required attrs except need-implied ones)."""
    return [a for a in spec.attrs if a not in spec.hidden_attrs]


def _place_abbrev(spec: Spec) -> str | None:
    if spec.district is not None:
        m = _QUAN_RE.match(spec.district)
        return f"q{m.group(1)}" if m else None
    if spec.city is not None:
        return CITY_ABBR.get(spec.city)
    return None


def _core_english(spec: Spec, rng: Random) -> str | None:
    """English core for the mixed degradation (only category+attr families)."""
    if spec.category is None or spec.category not in CAT_EN:
        return None
    prefixes = [EN_PREFIX_ATTR[a] for a in _render_attrs(spec) if a in EN_PREFIX_ATTR]
    suffixes = [EN_SUFFIX_ATTR[a] for a in _render_attrs(spec) if a in EN_SUFFIX_ATTR]
    if len(prefixes) + len(suffixes) != len(_render_attrs(spec)):
        return None  # an attribute lacks an EN rendering -> mixed not applicable
    cat_en = rng.choice(CAT_EN[spec.category])
    return " ".join([*prefixes, cat_en, *suffixes])


def _intent_word(spec: Spec, rng: Random) -> str | None:
    if spec.family == "superlative":
        return rng.choice(SUPERLATIVE_WORDS.get(spec.category, DEFAULT_SUPERLATIVE))
    if spec.family == "price":
        return rng.choice(PRICE_CHEAP if spec.price_dir == "cheap" else PRICE_UPSCALE)
    if spec.family == "open-late":
        return rng.choice(H247_WORDS if spec.hours_mode == "247" else LATE_WORDS)
    return None


@dataclass
class Rendered:
    spec: Spec
    core_vi: str          # Vietnamese pre-location text
    core_en: str | None   # English pre-location text (mixed) or None
    place_disp: str       # Vietnamese place display
    place_abbr: str | None
    intent_word: str | None
    place_canonical: str  # canonical district/city (for semantic_requirements)


def precompute(spec: Spec, rng: Random) -> Rendered:
    attr_phrases = [rng.choice(ATTR_QUERY[a]) for a in _render_attrs(spec)]
    intent_word = _intent_word(spec, rng)
    if spec.family == "brand":
        core_vi = spec.name_fragment or ""
    elif spec.family == "paraphrase":
        core_vi = " ".join([rng.choice(spec.need_phrases), *attr_phrases])
    else:
        cat_disp = rng.choice(CAT_QUERY[spec.category])
        bits = [cat_disp, *attr_phrases]
        if intent_word:
            bits.append(intent_word)
        core_vi = " ".join(bits)
    if spec.district is not None:
        place_disp = spec.district
        place_canonical = spec.district
    else:
        place_disp = rng.choice(CITY_DISP[spec.city])
        place_canonical = spec.city
    core_en = _core_english(spec, rng) if spec.family in {"attribute-seek", "category-location"} else None
    return Rendered(spec=spec, core_vi=core_vi, core_en=core_en, place_disp=place_disp,
                    place_abbr=_place_abbrev(spec), intent_word=intent_word,
                    place_canonical=place_canonical)


def applicable_styles(r: Rendered) -> dict[str, float]:
    styles = {k: v for k, v in STYLE_W.items() if k in {"none", "diacritics", "typo", "both"}}
    if r.place_abbr is not None:
        styles["abbrev"] = STYLE_W["abbrev"]
    if r.core_en is not None:
        styles["mixed"] = STYLE_W["mixed"]
    return styles


def render_text(r: Rendered, style: str, rng: Random) -> tuple[str, list[str]]:
    if style == "mixed" and r.core_en is not None:
        core, place = r.core_en, r.place_disp
    elif style == "abbrev" and r.place_abbr is not None:
        core, place = r.core_vi, r.place_abbr
    else:
        core, place = r.core_vi, r.place_disp
    text = re.sub(r"\s+", " ", f"{core} ở {place}").strip()
    if style in ("diacritics", "both"):
        text = strip_diacritics(text)
    flags = list(STYLE_FLAGS[style])
    if style in ("typo", "both"):
        text, applied = inject_typo(rng, text)
        if not applied:
            flags = [f for f in flags if f != "Typo"]
    return text, flags


# --------------------------------------------------------------------------- #
# Finalize a query into an EvalQuery-shaped row                                #
# --------------------------------------------------------------------------- #
def _bump(difficulty: str) -> str:
    return {"Easy": "Medium", "Medium": "Hard", "Hard": "Hard"}[difficulty]


def _difficulty(spec: Spec, style: str) -> str:
    """Family-driven (SPEC): Easy = name/brand or a clean category+district with strong
    signals; Medium = attribute-seek/category+attr; Hard = paraphrase, superlative, price,
    open-late. The heavier degradations (typo+diacritics together) bump one level."""
    if spec.family == "brand":
        base = "Easy"
    elif spec.family == "category-location":
        # A clean category+district (no tightening attribute needed) is a strong-signal Easy;
        # anything needing an attribute or scoped to a whole city is Medium.
        strong = spec.district is not None and not _render_attrs(spec)
        base = "Easy" if strong else "Medium"
    elif spec.family == "attribute-seek":
        base = "Medium"
    else:
        base = "Hard"
    return _bump(base) if style == "both" else base


def _semantic_requirements(spec: Spec, r: Rendered) -> list[str]:
    sem: list[str] = []
    if spec.family == "brand":
        sem.append(spec.name_fragment or "")
    else:
        sem.append(spec.category)
    sem.extend(spec.attrs)
    if spec.hours_mode == "late":
        sem.append("mở khuya")
    elif spec.hours_mode == "247":
        sem.append("24/7")
    if spec.order == "price_asc":
        sem.append("giá rẻ")
    elif spec.order == "price_desc":
        sem.append("cao cấp")
    elif spec.family == "superlative":
        sem.append("chất lượng cao")
    sem.append(r.place_canonical)
    return [s for s in sem if s]


@dataclass
class GenQuery:
    row: dict
    predicate: Callable[[POI], bool]
    family: str
    style: str


def finalize(spec: Spec, cands: list[POI], r: Rendered, style: str, flags: list[str],
             text: str, qid: str) -> GenQuery:
    ordered = sorted(cands, key=order_key(spec.order))[:HARD_CAP]
    skills = list(FAMILY_SKILLS[spec.family])
    for f in flags:
        if f not in skills:
            skills.append(f)
    qcat = "Mixed Language Search" if style == "mixed" else FAMILY_QCAT[spec.family]
    row = {
        "query_id": qid,
        "input_query": text,
        "query_category": qcat,
        "difficulty": _difficulty(spec, style),
        "expected_ids": [p.poi_id for p in ordered],
        "expected_names": [p.name for p in ordered],
        "expected_intent": FAMILY_INTENT[spec.family],
        "semantic_requirements": _semantic_requirements(spec, r),
        "ranking_signals": list(FAMILY_SIGNALS[spec.family]),
        "skills_tested": skills,
    }
    return GenQuery(row=row, predicate=predicate_for(spec), family=spec.family, style=style)


# --------------------------------------------------------------------------- #
# Top-level generation                                                         #
# --------------------------------------------------------------------------- #
def build_queries(pois: list[POI], seed: int, n: int) -> list[GenQuery]:
    """Deterministically build `n` labeled queries. Same (pois, seed, n) -> identical output."""
    rng = Random(seed)
    official_texts = {q.input_query for q in load_eval()}
    official_folded = {fold(q.input_query) for q in load_eval()}
    out: list[GenQuery] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = n * 120
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        family = weighted_choice(rng, FAMILY_W)
        built = BUILDERS[family](rng, pois)
        if built is None:
            continue
        spec, cands = built
        r = precompute(spec, rng)
        style = weighted_choice(rng, applicable_styles(r))
        text, flags = render_text(r, style, rng)
        if not text or text in seen or text in official_texts or fold(text) in official_folded:
            continue
        seen.add(text)
        qid = f"SQ{len(out) + 1:03d}"
        out.append(finalize(spec, cands, r, style, flags, text, qid))
    if len(out) < n:
        raise RuntimeError(f"only generated {len(out)}/{n} queries in {attempts} attempts")
    return out


def to_rows(queries: list[GenQuery]) -> list[dict]:
    return [q.row for q in queries]


# --------------------------------------------------------------------------- #
# Summary / CLI                                                                #
# --------------------------------------------------------------------------- #
def _counts(items) -> str:
    from collections import Counter
    return ", ".join(f"{k}={v}" for k, v in sorted(Counter(items).items()))


def print_summary(queries: list[GenQuery]) -> None:
    print(f"\ngenerated {len(queries)} labeled queries")
    print("  by difficulty :", _counts(q.row["difficulty"] for q in queries))
    print("  by family     :", _counts(q.family for q in queries))
    print("  by degradation:", _counts(q.style for q in queries))
    print("  by query_cat  :", _counts(q.row["query_category"] for q in queries))
    print("\nsample rows:")
    for q in queries[:5]:
        row = q.row
        print(f"  {row['query_id']} [{q.family}/{q.style}/{row['difficulty']}] "
              f"{row['input_query']!r}")
        print(f"        -> expected_ids={row['expected_ids']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic labeled synthetic-query generator")
    ap.add_argument("--dataset", type=Path, default=Path("data/synth/synth_dataset.xlsx"))
    ap.add_argument("--out", type=Path, default=Path("data/synth/eval_synth.json"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()

    pois = load_pois(args.dataset)
    queries = build_queries(pois, args.seed, args.n)
    rows = to_rows(queries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} — {len(rows)} queries, seed={args.seed}")
    print_summary(queries)


if __name__ == "__main__":
    main()
