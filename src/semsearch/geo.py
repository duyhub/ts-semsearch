"""Geo utilities + anchor resolution (SPEC §7).

Distance signal needs a resolved anchor ("gần hồ gươm" -> coords). We resolve
from a curated landmark gazetteer, then district centroids derived from the
dataset, then POI names. The hand gazetteer is deliberately small for now;
OV7 flags broadening it from an offline OSM/GeoNames extract before relying on
location metrics for the private eval.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from .data import POI, Anchor
from .normalize import fold

EARTH_KM = 6371.0

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

    def resolve(self, folded_text: str) -> Anchor | None:
        """Best-effort anchor from an already-folded query fragment (name keeps diacritics)."""
        for key, (lat, lon, disp) in self.landmarks.items():
            if key in folded_text:
                return Anchor(name=disp, lat=lat, lon=lon)
        for key, (lat, lon, disp) in self.districts.items():
            if key in folded_text:
                return Anchor(name=disp, lat=lat, lon=lon)
        for key, (lat, lon, disp) in self.poi_names.items():
            if key and key in folded_text:
                return Anchor(name=disp, lat=lat, lon=lon)
        return None
