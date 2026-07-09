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

# Curated landmarks (folded name -> (lat, lon)). Seeded for the four dataset
# cities; extend from an offline gazetteer (OV7).
LANDMARKS: dict[str, tuple[float, float]] = {
    "ho guom": (21.0287, 105.8524),
    "ho hoan kiem": (21.0287, 105.8524),
    "ho tay": (21.0587, 105.8190),
    "pho co": (21.0334, 105.8500),
    "cho ben thanh": (10.7723, 106.6980),
    "pho di bo nguyen hue": (10.7745, 106.7040),
    "ben thanh": (10.7723, 106.6980),
    "bien my khe": (16.0545, 108.2470),
    "cau rong": (16.0611, 108.2270),
    "ho xuan huong": (11.9416, 108.4383),
    "cho da lat": (11.9430, 108.4370),
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
        # district centroids (mean of member POIs) keyed by folded "district" and "district city"
        groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for p in pois:
            groups[fold(p.district)].append((p.lat, p.lon))
            groups[fold(f"{p.district} {p.city}")].append((p.lat, p.lon))
        self.districts = {
            k: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v))
            for k, v in groups.items()
        }
        self.poi_names = {fold(p.name): (p.lat, p.lon) for p in pois}

    def resolve(self, folded_text: str) -> Anchor | None:
        """Best-effort anchor from an already-folded query fragment."""
        for name, (lat, lon) in self.landmarks.items():
            if name in folded_text:
                return Anchor(name=name, lat=lat, lon=lon)
        for name, (lat, lon) in self.districts.items():
            if name in folded_text:
                return Anchor(name=name, lat=lat, lon=lon)
        for name, (lat, lon) in self.poi_names.items():
            if name and name in folded_text:
                return Anchor(name=name, lat=lat, lon=lon)
        return None
