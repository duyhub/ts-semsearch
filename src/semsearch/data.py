"""Data contracts and loaders (SPEC §2).

Loads the sponsor xlsx into typed frames. This is the ground truth the whole
pipeline iterates against, so parsing is strict and explicit.

Phase-0 verified facts baked in here (see SPEC §2):
  - poi_id is bare ("C001"); the API prepends "poi:" on output, never here.
  - `;`-separated: attributes, tags, expected_top_poi_ids, skills_tested.
  - comma-separated: expected_semantic_requirements, ranking_signals_to_use.
  - opening_hours: "HH:MM-HH:MM" | "24/7" | overnight ("18:00-03:00").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .normalize import fold

# Column name deltas between the xlsx and our dataclass (SPEC §2 Phase-0 note).
_POI_RENAME = {
    "poi_name": "name",
    "latitude": "lat",
    "longitude": "lon",
    "popularity_score": "popularity",
}


@dataclass
class Anchor:
    """A resolved location reference (POI / landmark / district / coordinate)."""

    name: str
    lat: float
    lon: float


@dataclass
class POI:
    poi_id: str
    name: str
    brand: str | None
    category: str
    sub_category: str | None
    city: str
    district: str
    address: str
    lat: float
    lon: float
    rating: float
    review_count: int
    popularity: float
    price_level: int | None
    opening_hours: str | None
    attributes: list[str]
    tags: list[str]
    description: str
    doc_text: str = ""  # composed embedding text (SPEC §4); filled at ingest


@dataclass
class QueryIntent:
    """Structured intent extracted from a raw query (SPEC §7). Populated in later phases."""

    raw: str
    normalized: str
    category: str | None = None
    anchor: Anchor | None = None
    required_attrs: list[str] = field(default_factory=list)
    soft_prefs: list[str] = field(default_factory=list)
    open_after: str | None = None
    price_max: int | None = None
    price_pref: str | None = None  # "cheap" | "expensive" | None — parsed affordability direction
    city: str | None = None
    district: str | None = None
    content_terms: list[str] = field(default_factory=list)  # distinctive leftover terms -> subject filter
    has_residual: bool = False  # any leftover content token -> category hard-filter ineligible
    residual_terms: list[str] = field(default_factory=list)  # ALL leftover terms (>= content_terms);
    # lets the pipeline tell a discredited subject from genuine unexplained content


@dataclass
class RankedResult:
    poi: POI
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


@dataclass
class EvalQuery:
    query_id: str
    input_query: str
    query_category: str
    difficulty: str
    expected_ids: list[str]  # relevance-ordered: first gain 3, second 2, rest 1 (SPEC §2)
    expected_names: list[str] = field(default_factory=list)
    expected_intent: str | None = None
    semantic_requirements: list[str] = field(default_factory=list)
    ranking_signals: list[str] = field(default_factory=list)
    skills_tested: list[str] = field(default_factory=list)


def content_tokens(poi: POI) -> set[str]:
    """Folded token set of a POI's *name + brand*. Used for the subject hard-filter
    (SPEC §6) and its corpus document-frequency. Deliberately narrow — a distinctive
    term must NAME the POI (e.g. "bún chả" in "Bún Chả Hương Liên"), not merely appear
    in its tags or free-text description; otherwise rare descriptors/amenities/verbs
    ("tối", "học", "pool", "ngoài trời", "món") would wrongly filter out valid results."""
    return {t for t in fold(f"{poi.name} {poi.brand or ''}").split() if t}


def _split(value: object, sep: str) -> list[str]:
    """Split a delimited cell into a clean list, dropping blanks. NaN/None -> []."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(sep) if part.strip()]


def _opt_str(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _opt_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


DEFAULT_XLSX = Path("data/raw/ai_maps_track2_dataset_participants.xlsx")


def load_pois(path: str | Path = DEFAULT_XLSX) -> list[POI]:
    """Load the POI_Dataset sheet into typed POI records."""
    df = pd.read_excel(path, sheet_name="POI_Dataset").rename(columns=_POI_RENAME)
    pois: list[POI] = []
    for row in df.to_dict(orient="records"):
        pois.append(
            POI(
                poi_id=str(row["poi_id"]).strip(),
                name=str(row["name"]).strip(),
                brand=_opt_str(row.get("brand")),
                category=str(row["category"]).strip(),
                sub_category=_opt_str(row.get("sub_category")),
                city=str(row["city"]).strip(),
                district=str(row["district"]).strip(),
                address=str(row["address"]).strip(),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                rating=float(row["rating"]),
                review_count=int(row["review_count"]),
                popularity=float(row["popularity"]),
                price_level=_opt_int(row.get("price_level")),
                opening_hours=_opt_str(row.get("opening_hours")),
                attributes=_split(row.get("attributes"), ";"),
                tags=_split(row.get("tags"), ";"),
                description=str(row.get("description") or "").strip(),
            )
        )
    return pois


def load_eval(path: str | Path = DEFAULT_XLSX) -> list[EvalQuery]:
    """Load the Public_Evaluation sheet into typed EvalQuery records."""
    df = pd.read_excel(path, sheet_name="Public_Evaluation")
    queries: list[EvalQuery] = []
    for row in df.to_dict(orient="records"):
        queries.append(
            EvalQuery(
                query_id=str(row["query_id"]).strip(),
                input_query=str(row["input_query"]).strip(),
                query_category=str(row["query_category"]).strip(),
                difficulty=str(row["difficulty"]).strip(),
                expected_ids=_split(row.get("expected_top_poi_ids"), ";"),
                expected_names=_split(row.get("expected_top_poi_names"), ";"),
                expected_intent=_opt_str(row.get("expected_intent")),
                semantic_requirements=_split(row.get("expected_semantic_requirements"), ","),
                ranking_signals=_split(row.get("ranking_signals_to_use"), ","),
                skills_tested=_split(row.get("skills_tested"), ";"),
            )
        )
    return queries
