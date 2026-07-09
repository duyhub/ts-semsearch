"""Interpretable 9-signal linear ranker (SPEC §6; PRD FR-7).

Six signals map 1:1 to the sponsor's Ranking_Signals; `category` and `price` are our
additions (category-consistency prior + affordability preference). Review fixes baked in:
  - semantic: fixed calibrated cosine band, NOT per-query min-max (OV6)
  - open_now: injected clock, handles 24/7 + overnight wraparound (A1, Phase-0)
  - rating: low Bayesian prior m so the narrow 3.8-4.7 band still varies (TODO-2)
Every result keeps a per-signal breakdown for explanations (FR-8).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .data import POI, QueryIntent
from .geo import haversine
from .normalize import fold

SIGNALS = ("semantic", "attributes", "category", "distance", "rating", "popularity",
           "open_now", "review", "price")

# Fixed cosine calibration band (OV6): a 0.30 cosine reads as 0, 0.75+ as 1.
COS_LO, COS_HI = 0.30, 0.75
RATING_M = 30.0          # low Bayesian prior (TODO-2)
RATING_LO, RATING_HI = 3.5, 5.0
DISTANCE_TAU_KM = 3.0    # exp(-d/tau) decay (SPEC §6)
PRICE_MIN, PRICE_MAX = 1, 4  # dataset price_level range
NEUTRAL = 0.5

# Committed reference time for eval (A1): deterministic, so open_now can't drift.
DEFAULT_EVAL_NOW = datetime(2026, 7, 11, 14, 0)  # 14:00, Asia/Ho_Chi_Minh assumed

# Default weights (pre-tuning). tune.py overwrites the 8 tunable signals in
# data/weights.json; `price` is a DELIBERATE FIXED preference weight (not eval-tuned —
# only 2/60 eval queries express price, too few to inform it; NFR-6). weights.json has
# no `price` key, so load_weights() supplies it here via its per-key fallback, leaving
# the proven tuned weights untouched.
DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.30, "attributes": 0.25, "category": 0.20, "distance": 0.10,
    "rating": 0.10, "popularity": 0.05, "open_now": 0.10, "review": 0.10,
    "price": 0.20,
}
WEIGHTS_PATH = Path("data/weights.json")  # committed, tuned on tune split only (NFR-6)


def load_weights(path: Path = WEIGHTS_PATH) -> dict[str, float]:
    """Tuned weights if present, else the defaults. Missing keys fall back to default."""
    if not path.exists():
        return dict(DEFAULT_WEIGHTS)
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh).get("weights", {})
    return {k: float(saved.get(k, DEFAULT_WEIGHTS[k])) for k in SIGNALS}

_HHMM = re.compile(r"^(\d{1,2}):(\d{2})-(\d{1,2}):(\d{2})$")


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def semantic_signal(cosine: float) -> float:
    return _clamp01((cosine - COS_LO) / (COS_HI - COS_LO))


def attributes_signal(intent: QueryIntent, poi_attrs_folded: set[str]) -> float:
    req = {fold(a) for a in intent.required_attrs}
    soft = {fold(a) for a in intent.soft_prefs}
    denom = len(req) + 0.5 * len(soft)
    if denom == 0:
        return NEUTRAL
    matched = len(req & poi_attrs_folded) + 0.5 * len(soft & poi_attrs_folded)
    return _clamp01(matched / denom)


def category_signal(intent: QueryIntent, poi: POI) -> float:
    """1.0 when the POI matches the query's parsed category, 0.0 otherwise;
    neutral (0.5) when no category was parsed so category-less queries
    (Discovery/Intent) are unaffected. A soft signal, not a hard filter — a
    parser mis-category only mildly penalizes the true answer instead of
    banishing it (SPEC §6; PRD FR-7)."""
    if intent.category is None:
        return NEUTRAL
    return 1.0 if poi.category == intent.category else 0.0


def distance_signal(intent: QueryIntent, poi: POI) -> float:
    if intent.anchor is None:
        return NEUTRAL
    d = haversine(intent.anchor.lat, intent.anchor.lon, poi.lat, poi.lon)
    return _clamp01(pow(2.718281828, -d / DISTANCE_TAU_KM))


def rating_signal(poi: POI, global_mean: float) -> float:
    v = poi.review_count
    bayes = (v / (v + RATING_M)) * poi.rating + (RATING_M / (v + RATING_M)) * global_mean
    return _clamp01((bayes - RATING_LO) / (RATING_HI - RATING_LO))


def popularity_signal(poi: POI) -> float:
    return _clamp01(poi.popularity / 100.0)


def _minutes(now: datetime) -> int:
    return now.hour * 60 + now.minute


def _is_open(hours: str | None, now_min: int) -> bool | None:
    """Open at now_min? None = unknown. Handles 24/7 and overnight wraparound (Phase-0)."""
    if not hours:
        return None
    if hours.strip() == "24/7":
        return True
    m = _HHMM.match(hours.strip())
    if not m:
        return None
    start = int(m[1]) * 60 + int(m[2])
    end = int(m[3]) * 60 + int(m[4])
    if end < start:  # crosses midnight, e.g. 18:00-03:00
        return now_min >= start or now_min <= end
    return start <= now_min <= end


def open_now_signal(intent: QueryIntent, poi: POI, now: datetime) -> float:
    # If the query wants late-night ("mở khuya" -> open_after), test at that hour.
    if intent.open_after:
        hh, _, mm = intent.open_after.partition(":")
        target = int(hh) * 60 + (int(mm) if mm else 0)
    else:
        target = _minutes(now)
    state = _is_open(poi.opening_hours, target)
    if state is None:
        return NEUTRAL
    return 1.0 if state else 0.3


def price_signal(intent: QueryIntent, poi: POI) -> float:
    """Affordability preference. NEUTRAL (0.5) when the query has no price intent —
    constant across POIs, so it leaves the ranking of price-less queries unchanged —
    and when a POI has no price_level. Otherwise a cheap intent scores cheaper POIs
    high (level 1 -> 1.0), an expensive intent inverts it. Bidirectional, unlike the
    always-higher-is-better signals (SPEC §6; PRD FR-7)."""
    if intent.price_pref is None or poi.price_level is None:
        return NEUTRAL
    norm = (poi.price_level - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)  # 0=cheapest .. 1=priciest
    return _clamp01(1.0 - norm if intent.price_pref == "cheap" else norm)


def review_signal(intent: QueryIntent, poi_review_tokens: set[str]) -> float:
    """Query need-terms matched against POI tags + description (distinct from the
    structured attributes field, SPEC §6)."""
    q = set(intent.normalized.split())
    if not q or not poi_review_tokens:
        return NEUTRAL
    return _clamp01(len(q & poi_review_tokens) / len(q))


@dataclass
class LinearRanker:
    weights: dict[str, float]
    now: datetime
    global_rating_mean: float

    def signals(self, relevance: float, intent: QueryIntent, poi: POI,
                attrs_folded: set[str], review_tokens: set[str]) -> dict[str, float]:
        """`relevance` is the pre-calibrated retrieval relevance in [0,1]
        (hybrid RRF from the pipeline). Seeding semantic from hybrid means the
        ranker starts from hybrid strength and can only add to it (full >= hybrid)."""
        return {
            "semantic": _clamp01(relevance),
            "attributes": attributes_signal(intent, attrs_folded),
            "category": category_signal(intent, poi),
            "distance": distance_signal(intent, poi),
            "rating": rating_signal(poi, self.global_rating_mean),
            "popularity": popularity_signal(poi),
            "open_now": open_now_signal(intent, poi, self.now),
            "review": review_signal(intent, review_tokens),
            "price": price_signal(intent, poi),
        }

    def score(self, relevance: float, intent: QueryIntent, poi: POI,
              attrs_folded: set[str], review_tokens: set[str]) -> tuple[float, dict[str, float]]:
        b = self.signals(relevance, intent, poi, attrs_folded, review_tokens)
        total_w = sum(self.weights.values()) or 1.0
        s = sum(self.weights.get(k, 0.0) * b[k] for k in SIGNALS) / total_w
        return s, b
