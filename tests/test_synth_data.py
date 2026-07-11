"""Synthetic-corpus generator guards (Worker D).

Proves scripts/gen_synth_pois.py produces a deterministic 1000-POI stress corpus that
`semsearch.data.load_pois` reads UNCHANGED: the official 111 rows verbatim + 889 seeded,
realistic distractors constrained to the dataset's own geography/category/attribute vocab.

Generation is <1s, so we build fresh into tmp_path (module-scoped) rather than trust a
committed artifact — the guards then hold on any clean checkout.
"""
from __future__ import annotations

import importlib.util
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from semsearch.data import load_pois

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "data" / "raw" / "ai_maps_track2_dataset_participants.xlsx"
SEED = 20260711
N_TOTAL = 1000
N_OFFICIAL = 111
N_SYNTH = N_TOTAL - N_OFFICIAL

# Vietnamese diacritic characters (precomposed) + đ/Đ — the marker of Vietnamese text.
_VIET = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_VIET_CHARS = set(_VIET + _VIET.upper())
_SYN_ID = re.compile(r"^SYN\d{4}$")


def _has_diacritic(text: str) -> bool:
    return any(c in _VIET_CHARS for c in text)


def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "gen_synth_pois", ROOT / "scripts" / "gen_synth_pois.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def out_path(gen, tmp_path_factory):
    out = tmp_path_factory.mktemp("synth") / "synth_dataset.xlsx"
    df_poi, df_tax = gen.build_dataframe(OFFICIAL, N_TOTAL, SEED)
    gen.write_xlsx(out, df_poi, df_tax)
    return out


@pytest.fixture(scope="module")
def official_pois():
    return load_pois(OFFICIAL)


@pytest.fixture(scope="module")
def taxonomy_attrs():
    df = pd.read_excel(OFFICIAL, sheet_name="Attribute_Taxonomy")
    return {str(a).strip() for a in df["attribute"]}


@pytest.fixture(scope="module")
def pois(out_path):
    return load_pois(out_path)


@pytest.fixture(scope="module")
def synth(pois):
    return pois[N_OFFICIAL:]


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_determinism_same_seed_identical(gen, tmp_path):
    """Two runs at the same seed produce byte-identical LOGICAL content (rows + order)."""
    a, b = tmp_path / "a.xlsx", tmp_path / "b.xlsx"
    for out in (a, b):
        df_poi, df_tax = gen.build_dataframe(OFFICIAL, N_TOTAL, SEED)
        gen.write_xlsx(out, df_poi, df_tax)
    import dataclasses

    rows_a = [dataclasses.astuple(p) for p in load_pois(a)]
    rows_b = [dataclasses.astuple(p) for p in load_pois(b)]
    assert rows_a == rows_b


# --------------------------------------------------------------------------- #
# superset: official rows preserved verbatim
# --------------------------------------------------------------------------- #
def test_first_111_ids_match_official(pois, official_pois):
    assert [p.poi_id for p in pois[:N_OFFICIAL]] == [p.poi_id for p in official_pois]


def test_official_field_values_verbatim(pois, official_pois):
    """Every field of the first 111 rows matches the official dataset exactly."""
    import dataclasses

    got = [dataclasses.astuple(p) for p in pois[:N_OFFICIAL]]
    want = [dataclasses.astuple(p) for p in official_pois]
    assert got == want


# --------------------------------------------------------------------------- #
# schema: loader reads exactly 1000 valid POIs within the dataset's vocab
# --------------------------------------------------------------------------- #
def test_loads_exactly_1000(pois):
    assert len(pois) == N_TOTAL


def test_ids_unique_and_syn_well_formed(pois, synth):
    assert len({p.poi_id for p in pois}) == N_TOTAL
    assert [p.poi_id for p in synth] == [f"SYN{i:04d}" for i in range(1, N_SYNTH + 1)]
    assert all(_SYN_ID.match(p.poi_id) for p in synth)


def test_synth_categories_in_official_set(synth, official_pois):
    official_cats = {p.category for p in official_pois}
    assert {p.category for p in synth} <= official_cats


def test_synth_attributes_within_taxonomy(synth, taxonomy_attrs):
    """Synthetic attributes are drawn ONLY from the closed 10-attribute taxonomy.
    (Official rows keep their free-form attributes — hence this is scoped to SYN rows.)"""
    assert len(taxonomy_attrs) == 10
    offenders = {a for p in synth for a in p.attributes if a not in taxonomy_attrs}
    assert not offenders, f"synthetic attrs outside taxonomy: {offenders}"


def test_synth_cities_and_districts_in_official(synth, official_pois):
    cities = {p.city for p in official_pois}
    pairs = {(p.city, p.district) for p in official_pois}
    assert {p.city for p in synth} <= cities
    assert {(p.city, p.district) for p in synth} <= pairs


def test_synth_latlon_within_city_box(synth, official_pois):
    boxes: dict[str, list[float]] = {}
    for p in official_pois:
        b = boxes.setdefault(p.city, [p.lat, p.lat, p.lon, p.lon])
        b[0], b[1] = min(b[0], p.lat), max(b[1], p.lat)
        b[2], b[3] = min(b[2], p.lon), max(b[3], p.lon)
    for p in synth:
        lo_lat, hi_lat, lo_lon, hi_lon = boxes[p.city]
        assert lo_lat <= p.lat <= hi_lat and lo_lon <= p.lon <= hi_lon, p.poi_id


def test_synth_attr_count_in_range(synth):
    assert all(2 <= len(p.attributes) <= 5 for p in synth)


def test_coastal_attribute_only_in_coastal_city(synth):
    """'gần biển' (near beach) is never assigned to an inland city — no geo contradictions."""
    assert {p.city for p in synth if "gần biển" in p.attributes} <= {"Đà Nẵng"}


# --------------------------------------------------------------------------- #
# diacritics: synthetic Vietnamese text carries diacritics
# --------------------------------------------------------------------------- #
def test_diacritics_coverage(synth):
    hits = sum(_has_diacritic(f"{p.name} {p.description}") for p in synth)
    assert hits / len(synth) >= 0.95


# --------------------------------------------------------------------------- #
# distribution sanity
# --------------------------------------------------------------------------- #
def test_every_official_category_has_min_synth_members(synth, official_pois):
    official_cats = {p.category for p in official_pois}
    counts = Counter(p.category for p in synth)
    for cat in official_cats:
        assert counts[cat] >= 10, f"{cat} has only {counts[cat]} synthetic members"


def test_no_category_dominates(pois):
    counts = Counter(p.category for p in pois)
    top_share = max(counts.values()) / len(pois)
    assert top_share <= 0.40, f"a category is {top_share:.0%} of the corpus"
