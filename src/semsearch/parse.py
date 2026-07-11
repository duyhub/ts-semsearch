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
from .geo import Gazetteer, detect_coordinate_anchor
from .normalize import (
    STOPWORDS,
    canonicalize,
    common_word_folds,
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

# Need keywords: DISPLAY (diacritic) NEED phrase -> canonical dataset category. A
# conversational need-query ("mình đói bụng quá" = I'm hungry) names no place-type word,
# so CATEGORY_KEYWORDS never fires and the category hard-filter stays disabled — the
# ranking then collapses onto noisy semantic+popularity (a 24/7 gas station floats up).
# This lexicon infers the category from the *need*; it is consulted ONLY when no explicit
# CATEGORY_KEYWORDS match fired (place-type words always win), and matched need tokens are
# consumed exactly like category-keyword tokens so a fully-explained need query has empty
# residual (which is what re-enables the pipeline's category hard-filter).
#
# FOLD-AMBIGUITY: every key is a >=2-token phrase whenever its lead token folds onto an
# unrelated word — "đói bụng" -> "doi bung" is safe as a bigram, but a bare "đói"/"doi"
# would collide with đôi/đối/đồi, so it is NEVER added alone; the ONLY doi-based trigger is
# the "đói bụng" bigram. Values reuse the exact 12 dataset category strings (no new values).
# Phrases already covered by CATEGORY_KEYWORDS (rút tiền/ATM, hiệu thuốc/Nhà thuốc,
# tham quan/Điểm tham quan) are intentionally omitted — the explicit pass handles them.
NEED_KEYWORDS: dict[str, str] = {
    # hunger / eating -> restaurant
    "đói bụng": "Nhà hàng", "muốn ăn": "Nhà hàng", "ăn gì": "Nhà hàng",
    "chỗ ăn": "Nhà hàng", "ăn uống": "Nhà hàng", "ăn ngon": "Nhà hàng",
    "ăn trưa": "Nhà hàng", "ăn tối": "Nhà hàng", "bữa trưa": "Nhà hàng",
    "bữa tối": "Nhà hàng",
    # thirst / rest -> café
    "khát nước": "Quán cà phê", "khát quá": "Quán cà phê", "muốn uống": "Quán cà phê",
    "uống gì": "Quán cà phê", "giải khát": "Quán cà phê", "nghỉ chân": "Quán cà phê",
    # fuel -> gas station
    "đổ xăng": "Trạm xăng", "hết xăng": "Trạm xăng", "bơm xăng": "Trạm xăng",
    # medicine / illness -> pharmacy
    "mua thuốc": "Nhà thuốc", "thuốc cảm": "Nhà thuốc", "bị ốm": "Nhà thuốc",
    "bị cảm": "Nhà thuốc",
    # movies -> cinema
    "xem phim": "Rạp phim", "coi phim": "Rạp phim",
    # medical care -> hospital
    "khám bệnh": "Bệnh viện", "cấp cứu": "Bệnh viện", "đi khám": "Bệnh viện",
    # shopping -> mall
    "mua sắm": "Trung tâm thương mại", "đi mua đồ": "Trung tâm thương mại",
    # EV charging -> charging station
    "sạc xe điện": "Trạm sạc điện", "hết pin xe": "Trạm sạc điện",
    "sạc ô tô điện": "Trạm sạc điện",
    # lodging -> hotel
    "qua đêm": "Khách sạn", "đặt phòng": "Khách sạn", "nghỉ qua đêm": "Khách sạn",
    "chỗ ngủ": "Khách sạn",
    # strolling / fresh air -> park
    "đi dạo": "Công viên", "hóng mát": "Công viên",
    # sightseeing / outings -> attraction
    "chỗ chơi": "Điểm tham quan", "đi chơi ở đâu": "Điểm tham quan",
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
        self._need_keys = sorted(NEED_KEYWORDS, key=lambda k: len(fold(k)), reverse=True)
        self._attr_keys = sorted(ATTRIBUTE_KEYWORDS, key=lambda k: len(fold(k)), reverse=True)
        # corpus document-frequency of subject tokens, for distinctive-term detection
        self._df: Counter[str] = Counter()
        for p in pois:
            self._df.update(content_tokens(p))

        # --- Typo-correction vocabularies (PRD FR-3; SPEC §3) --------------------
        # TARGET = the closed structured vocabularies a typo may be corrected TOWARD:
        # category keyword tokens, attribute keyword tokens, and gazetteer LOCATION
        # names (cities, curated landmarks, district names). Multi-word keys
        # contribute their individual folded tokens (canonicalize matches token-wise).
        target: set[str] = set()
        for kw in CATEGORY_KEYWORDS:
            target.update(fold(kw).split())
        for kw in ATTRIBUTE_KEYWORDS:
            target.update(fold(kw).split())
        for fkey in self.city_vocab:  # folded city forms (e.g. "tp hcm")
            target.update(fkey.split())
        for _lat, _lon, disp in gazetteer.landmarks.values():
            target.update(fold(disp).split())
        for _lat, _lon, disp in gazetteer.districts.values():
            target.update(fold(disp).split())
        self._correct_target: frozenset[str] = frozenset(target)
        # KNOWN = every folded token the parser already RECOGNIZES, hence "not a typo":
        # the correction targets PLUS POI-name tokens (so a distinctive subject like
        # 'Chay'/'Quảng'/'Bạch' — a real POI-name word one edit from a vocab token — is
        # never force-corrected, SPEC §6 subject preservation), stopwords, and the
        # vendored Vietnamese common words (so 'phòng'/'trên'/'đoàn' stay put). Fuzzy
        # correction fires ONLY for out-of-vocabulary tokens (len >= 4) not in KNOWN.
        known: set[str] = set(target)
        for _lat, _lon, disp in gazetteer.poi_names.values():
            known.update(fold(disp).split())
        known.update(STOPWORDS)
        known.update(common_word_folds())
        self._correct_known: frozenset[str] = frozenset(known)

    def _typo_corrections(self, *token_lists: Sequence[str]) -> dict[str, str]:
        """Map each OOV query token to its unique edit-1 canonical form (PRD FR-3).

        A token is a correction candidate only if it is len >= 4 AND not already a
        RECOGNIZED word (self._correct_known — vocab/POI/stopword/common). Candidates
        are fuzzy-matched against the structured TARGET vocabulary via canonicalize
        (unique edit-1, transposition-preferring, ambiguity-refusing). The returned
        map is applied to the match haystack only — never to the echo/subject fields.
        """
        corr: dict[str, str] = {}
        seen: set[str] = set()
        for toks in token_lists:
            for t in toks:
                if t in seen:
                    continue
                seen.add(t)
                if len(t) < 4 or t in self._correct_known:
                    continue
                c = canonicalize(t, self._correct_target)
                if c is not None and c != t:
                    corr[t] = c
        return corr

    def parse(self, text: str) -> QueryIntent:
        folded = fold(text)
        folded_tokens = folded.split()
        exp_tokens = expand_query(text)
        expanded = " ".join(exp_tokens)  # echo field: keeps ORIGINAL (typo'd) tokens
        # Fuzzy-correct OOV typo tokens onto the closed vocabularies (PRD FR-3; SPEC §3).
        # The correction feeds category/attribute/location matching ONLY: it rewrites the
        # haystack but NOT intent.raw/normalized, and the residual/subject derivation below
        # works off the ORIGINAL tokens, so a distinctive subject is never rewritten into a
        # vocab word ('cafe yen tihn' -> attribute yên tĩnh; a real subject like 'bạch' is
        # left untouched because it is a recognized word).
        corr = self._typo_corrections(exp_tokens, folded_tokens)
        exp_corrected = " ".join(corr.get(t, t) for t in exp_tokens)
        folded_corrected = " ".join(corr.get(t, t) for t in folded_tokens)
        hay = f" {exp_corrected} {folded_corrected} "
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

        # Need inference: a conversational need ("đói bụng") fills the category ONLY when
        # no explicit place-type word already did (explicit always wins). Matched need
        # tokens are consumed like category-keyword tokens, so the need query ends with an
        # empty residual and the pipeline category hard-filter (SPEC §6) fires. Skipped
        # entirely on an explicit match, so a compound query's leftover need words (e.g.
        # "mua sắm" in a "... nhà hàng ..." query) still block the filter as before.
        if category is None:
            for key in self._need_keys:
                if token_key_matches(hay, text, key):
                    cand = NEED_KEYWORDS[key]
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
        # An explicit decimal lat/lon pair (SPEC §7.1; PRD FR-2) is the strongest
        # location signal, so it takes PRECEDENCE over gazetteer name resolution.
        # Detected on the RAW query before folding (fold() would shatter '10.7738'
        # into '10'/'7738'). The coordinate integer shards are pure digits, so they
        # are already excluded from residual/content_terms below (not t.isdigit() /
        # t.isalpha()); no extra consumption is needed.
        anchor = detect_coordinate_anchor(text)
        if anchor is None:
            # Resolve against the expanded haystack (so "q1" -> "quan 1" resolves, FR-2)
            # plus the raw query for diacritic compatibility (Fix 1: 'phở có' won't
            # anchor to Phố Cổ).
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
            # a corrected token counts as explained if its CANONICAL form was consumed,
            # so a typo the parse resolved ('tihn' -> 'tinh') never leaks into residual —
            # while an uncorrected token (corr.get(t, t) == t) behaves exactly as before.
            if t not in consumed and corr.get(t, t) not in consumed
            and t not in STOPWORDS and t not in common
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
