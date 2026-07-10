"""Geo utilities + anchor resolution (SPEC §7).

Distance signal needs a resolved anchor ("gần hồ gươm" -> coords). We resolve
from a curated landmark gazetteer, then district centroids derived from the
dataset, then POI names. The hand gazetteer is deliberately small for now;
OV7 flags broadening it from an offline OSM/GeoNames extract before relying on
location metrics for the private eval.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Sequence

from .data import POI, Anchor
from .normalize import compat_token_seq, fold, token_key_matches

EARTH_KM = 6371.0

# --- Coordinate-in-query detection (SPEC §7.1; PRD FR-2) ---
# Honest, natural-reading Vietnamese label: "cách vị trí đã chọn 1.2 km" reads as
# "1.2 km from the selected location" (see explain.py). Kept here so callers don't
# hardcode the string.
COORD_ANCHOR_NAME = "vị trí đã chọn"

# A decimal number: optional sign, integer part, a literal '.', 1+ fraction digits.
# The '.' is REQUIRED so a bare integer (a district number, "24" in "24/7") never
# reads as a coordinate — only an explicit decimal pair does.
_DECIMAL_RE = re.compile(r"-?\d+\.\d+")
# Two coordinates must be separated by ONLY a comma and/or whitespace (a bare hyphen
# is a range like "10-106", not a coordinate delimiter).
_COORD_SEP_RE = re.compile(r"[\s,]+")

# Vietnam WGS84 sanity bounds (SPEC §7.1): latitude 8–24, longitude 102–110.
_VN_LAT = (8.0, 24.0)
_VN_LON = (102.0, 110.0)


def _bounded_latlon(a: float, b: float) -> tuple[float, float] | None:
    """Return (lat, lon) if the pair sits inside Vietnam, else None.

    Tries the natural (lat, lon) order first; accepts the reversed (lon, lat) paste
    order ONLY when the natural order fails the bounds but the swap passes — so an
    unambiguous in-order pair is never silently transposed.
    """
    if _VN_LAT[0] <= a <= _VN_LAT[1] and _VN_LON[0] <= b <= _VN_LON[1]:
        return a, b
    if _VN_LAT[0] <= b <= _VN_LAT[1] and _VN_LON[0] <= a <= _VN_LON[1]:
        return b, a
    return None


def detect_coordinate_anchor(raw_text: str) -> Anchor | None:
    """Anchor from an explicit decimal lat/lon pair in the RAW query (SPEC §7.1; PRD FR-2).

    MUST run on raw text BEFORE folding: fold() turns '.' into whitespace, so
    '10.7738' would shatter into the tokens '10'/'7738' and no pair could be seen.
    Scans adjacent decimal numbers separated only by a comma and/or whitespace and
    returns the first pair that is sanity-bounded to Vietnam (lat 8–24, lon 102–110),
    honoring a reversed (lon, lat) paste order per ``_bounded_latlon``. Returns None
    when no in-bounds pair exists — a lone rating decimal ('3.5 sao'), a hyphen range,
    or an out-of-country pair all fall through (to gazetteer resolution / no anchor).
    """
    nums = list(_DECIMAL_RE.finditer(raw_text))
    for left, right in zip(nums, nums[1:]):
        gap = raw_text[left.end():right.start()]
        if not _COORD_SEP_RE.fullmatch(gap):  # separator must be comma/whitespace only
            continue
        latlon = _bounded_latlon(float(left.group()), float(right.group()))
        if latlon is not None:
            return Anchor(name=COORD_ANCHOR_NAME, lat=latlon[0], lon=latlon[1])
    return None

# Curated landmarks (folded name -> (lat, lon, display)). Seeded for the four
# dataset cities; extend from an offline gazetteer (OV7). Display keeps diacritics.
LANDMARKS: dict[str, tuple[float, float, str]] = {
    "ho guom": (21.0287, 105.8524, "Hồ Gươm"),
    "ho hoan kiem": (21.0287, 105.8524, "Hồ Hoàn Kiếm"),
    "ho tay": (21.0587, 105.8190, "Hồ Tây"),
    "pho co": (21.0334, 105.8500, "Phố Cổ"),
    "cho ben thanh": (10.7723, 106.6980, "Chợ Bến Thành"),
    "pho di bo nguyen hue": (10.7745, 106.7040, "Phố đi bộ Nguyễn Huệ"),
    "ben thanh": (10.7723, 106.6980, "Bến Thành"),
    "bien my khe": (16.0545, 108.2470, "Biển Mỹ Khê"),
    "cau rong": (16.0611, 108.2270, "Cầu Rồng"),
    "ho xuan huong": (11.9416, 108.4383, "Hồ Xuân Hương"),
    "cho da lat": (11.9430, 108.4370, "Chợ Đà Lạt"),
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_KM * math.asin(math.sqrt(a))


class Gazetteer:
    """Resolves a folded place name to coordinates: landmark -> district centroid -> POI name."""

    def __init__(self, pois: Sequence[POI]):
        self.landmarks = dict(LANDMARKS)
        # district centroids (mean of member POIs), keeping a diacritic display name
        groups: dict[str, tuple[list[tuple[float, float]], str]] = {}
        for p in pois:
            for key, disp in ((fold(p.district), p.district), (fold(f"{p.district} {p.city}"), f"{p.district}, {p.city}")):
                pts, _ = groups.setdefault(key, ([], disp))
                pts.append((p.lat, p.lon))
        self.districts = {
            k: (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts), disp)
            for k, (pts, disp) in groups.items()
        }
        self.poi_names = {fold(p.name): (p.lat, p.lon, p.name) for p in pois}

    def resolve(self, folded_text: str, raw_text: str | None = None) -> Anchor | None:
        """Best-effort anchor from a folded query fragment plus the RAW query.

        Matching is diacritic-compatible and token-boundary (Fix 1): landmark and
        POI-name keys are checked against the raw query directly, while district
        keys are checked against the (possibly abbreviation-expanded) folded
        haystack — so 'q1' -> 'quan 1' still resolves — but rejected if the raw
        query spells a conflicting diacritic. ``raw_text`` defaults to
        ``folded_text`` for callers that only hold folded input (already
        diacritic-free, hence permissive). Names keep their diacritics.
        """
        raw = folded_text if raw_text is None else raw_text
        for _key, (lat, lon, disp) in self.landmarks.items():
            if compat_token_seq(raw, disp):
                return Anchor(name=disp, lat=lat, lon=lon)
        for _key, (lat, lon, disp) in self.districts.items():
            if token_key_matches(folded_text, raw, disp):
                return Anchor(name=disp, lat=lat, lon=lon)
        for _key, (lat, lon, disp) in self.poi_names.items():
            if disp and compat_token_seq(raw, disp):
                return Anchor(name=disp, lat=lat, lon=lon)
        return None
