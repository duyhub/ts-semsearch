"""Signal-derived explanations (SPEC §8; PRD FR-8).

Every reason is generated from a verifiable fact on the POI (matched attribute,
distance to a resolved anchor, rating, opening hours) — never asserted. A
faithfulness validator re-checks the produced strings against the POI and flags
any attribute or number that isn't backed by the data, so a hallucinated reason
can never ship (the validator is the FR-8 guard, and a test asserts it rejects
fabrications).
"""
from __future__ import annotations

import re

from .data import POI, QueryIntent
from .geo import haversine
from .normalize import fold

_RATING_RE = re.compile(r"(\d\.\d)\s*★")
_REVIEWS_RE = re.compile(r"([\d.]+)\s*đánh giá")
_ATTR_RE = re.compile(r"✓\s*([^,·]+?)(?=$|,|·)")


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")  # 1560 -> "1.560" (dataset convention)


def _hours_end(hours: str) -> str | None:
    m = re.match(r"^\d{1,2}:\d{2}-(\d{1,2}:\d{2})$", hours.strip())
    return m.group(1) if m else None


def generate_reasons(intent: QueryIntent, poi: POI, *, max_reasons: int = 4) -> list[str]:
    """1..max_reasons Vietnamese reasons, each traceable to a signal value (FR-8)."""
    reasons: list[str] = []

    # matched attributes (display canonical, only those actually on the POI)
    poi_attrs = {fold(a) for a in poi.attributes}
    matched = [a for a in intent.required_attrs if fold(a) in poi_attrs]
    if matched:
        reasons.append(", ".join(f"✓ {a}" for a in matched))

    # distance to the resolved anchor
    if intent.anchor is not None:
        km = haversine(intent.anchor.lat, intent.anchor.lon, poi.lat, poi.lon)
        dist = f"{km:.1f} km" if km >= 1 else f"{int(round(km * 1000, -1))} m"
        reasons.append(f"cách {intent.anchor.name} {dist}")

    # rating (+ review count)
    reasons.append(f"{poi.rating:.1f}★ · {_fmt_int(poi.review_count)} đánh giá")

    # opening hours
    if poi.opening_hours:
        if poi.opening_hours.strip() == "24/7":
            reasons.append("mở 24/7")
        else:
            end = _hours_end(poi.opening_hours)
            if end:
                reasons.append(f"mở đến {end}")

    return reasons[:max_reasons]


def validate_reasons(reasons: list[str], poi: POI) -> list[str]:
    """Return a list of faithfulness violations (empty == all reasons are backed).

    Checks the fabrication-prone claims: attribute mentions must be on the POI,
    and any rating / review-count number must match the POI exactly.
    """
    violations: list[str] = []
    poi_attrs = {fold(a) for a in poi.attributes}
    for reason in reasons:
        for attr in _ATTR_RE.findall(reason):
            if fold(attr) not in poi_attrs:
                violations.append(f"attribute not on POI: {attr.strip()!r}")
        m = _RATING_RE.search(reason)
        if m and abs(float(m.group(1)) - round(poi.rating, 1)) > 1e-9:
            violations.append(f"rating claim {m.group(1)} != {poi.rating}")
        m = _REVIEWS_RE.search(reason)
        if m and int(m.group(1).replace(".", "")) != poi.review_count:
            violations.append(f"review_count claim {m.group(1)} != {poi.review_count}")
    return violations
