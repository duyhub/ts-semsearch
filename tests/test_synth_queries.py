"""Guards for the labeled synthetic eval-query generator (Worker E).

Proves scripts/gen_synth_queries.py produces a deterministic ~150-query stress eval set whose
ground truth is BY CONSTRUCTION — labels satisfy explicit predicates over raw POI fields, never
the search engine under test. The committed artifact data/synth/eval_synth.json must stay in
sync with a fresh regeneration, load cleanly as EvalQuery rows, and carry the degradation mix.

Generation is ~1s, so we regenerate in-memory (for determinism + predicate re-verification) and
also validate the committed JSON artifact. Nothing here imports pipeline/retrieve/rank/engines.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from semsearch.data import EvalQuery, load_eval, load_pois

ROOT = Path(__file__).resolve().parents[1]
SYNTH_XLSX = ROOT / "data" / "synth" / "synth_dataset.xlsx"
ARTIFACT = ROOT / "data" / "synth" / "eval_synth.json"
SEED = 20260711
N = 150

_SQ_ID = re.compile(r"^SQ\d{3}$")
_VIET = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
_VIET_CHARS = set(_VIET + _VIET.upper())
_DEGRADE_FLAGS = {"Diacritics", "Typo", "Abbreviation", "MixedLanguage"}


def _has_diacritic(text: str) -> bool:
    return any(c in _VIET_CHARS for c in text)


def _load_gen():
    """Import the generator by path. The module MUST be registered in sys.modules before
    exec — its @dataclass definitions resolve their __module__ there (else AttributeError)."""
    spec = importlib.util.spec_from_file_location(
        "gen_synth_queries", ROOT / "scripts" / "gen_synth_queries.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen():
    return _load_gen()


@pytest.fixture(scope="module")
def pois():
    return load_pois(SYNTH_XLSX)


@pytest.fixture(scope="module")
def by_id(pois):
    return {p.poi_id: p for p in pois}


@pytest.fixture(scope="module")
def rows():
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def records(gen, pois):
    """In-memory GenQuery records — each carries its constraint predicate for re-verification."""
    return gen.build_queries(pois, SEED, N)


# --------------------------------------------------------------------------- #
# determinism + artifact freshness
# --------------------------------------------------------------------------- #
def test_determinism_same_seed_identical(gen, pois):
    a = gen.to_rows(gen.build_queries(pois, SEED, N))
    b = gen.to_rows(gen.build_queries(pois, SEED, N))
    assert a == b


def test_committed_artifact_matches_regeneration(gen, records, rows):
    """The checked-in JSON is exactly what the default seed produces (script-generated, in sync)."""
    assert gen.to_rows(records) == rows


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_every_row_constructs_evalquery(rows):
    field_names = {f.name for f in dataclasses.fields(EvalQuery)}
    for row in rows:
        assert set(row) == field_names, f"row keys must match EvalQuery exactly: {row.get('query_id')}"
        EvalQuery(**row)  # must not raise


def test_ids_unique_and_well_formed(rows):
    assert 140 <= len(rows) <= 160
    ids = [r["query_id"] for r in rows]
    assert all(_SQ_ID.match(i) for i in ids), "ids must be SQ### (provenance)"
    assert len(set(ids)) == len(ids), "query ids must be unique"
    assert ids == [f"SQ{i:03d}" for i in range(1, len(rows) + 1)], "ids must be sequential"


def test_input_queries_unique(rows):
    qs = [r["input_query"] for r in rows]
    assert len(set(qs)) == len(qs)


# --------------------------------------------------------------------------- #
# label validity — ground truth is real and constrained
# --------------------------------------------------------------------------- #
def test_expected_ids_valid(rows, by_id):
    for r in rows:
        ids = r["expected_ids"]
        assert ids, f"{r['query_id']} has empty expected_ids"
        assert len(ids) <= 8, f"{r['query_id']} labels {len(ids)} > 8 POIs (too generic)"
        assert len(set(ids)) == len(ids), f"{r['query_id']} has duplicate expected ids"
        for pid in ids:
            assert pid in by_id, f"{r['query_id']} references unknown POI {pid}"
        assert r["expected_names"] == [by_id[p].name for p in ids]


def test_every_expected_poi_satisfies_its_predicate(records, by_id):
    """The core anti-circularity guarantee: labels come from explicit predicates over raw
    POI fields, so every expected POI (across ALL families) must satisfy its query's predicate."""
    for rec in records:
        for pid in rec.row["expected_ids"]:
            assert rec.predicate(by_id[pid]), f"{rec.row['query_id']} top/label {pid} fails predicate"


def test_expected_ids_are_the_complete_corpus_match(records, pois):
    """Labels = ALL corpus POIs satisfying the predicate (ordered, capped at 8) — nothing valid
    is silently dropped, nothing invalid is added."""
    for rec in records:
        matches = {p.poi_id for p in pois if rec.predicate(p)}
        assert len(matches) <= 8, f"{rec.row['query_id']} predicate matches {len(matches)} (> cap)"
        assert matches == set(rec.row["expected_ids"])


def test_label_validity_reverified_from_json(rows, by_id, gen):
    """INDEPENDENT re-check straight from the artifact (not the generator's closures): re-derive
    each query's attribute/category/location constraints from semantic_requirements and confirm the
    TOP expected POI satisfies them. Hours-based (Time) and brand (POI) queries are excluded — their
    constraint isn't attribute membership. Covers attribute-seek, category+loc, paraphrase, price,
    superlative predicate families."""
    taxo = set(gen.ATTR_QUERY)
    checked = 0
    for r in rows:
        if "Time" in r["skills_tested"] or r["query_category"] == "POI Search":
            continue
        top = by_id[r["expected_ids"][0]]
        sem = r["semantic_requirements"]
        # sem[0] is the canonical category for non-brand queries.
        assert top.category == sem[0], f"{r['query_id']} top category {top.category} != {sem[0]}"
        # every taxonomy attribute named in sem must be a real membership constraint on the top POI.
        for a in (s for s in sem if s in taxo):
            assert a in top.attributes, f"{r['query_id']} top POI lacks required attr {a!r}"
        # the last sem entry is the canonical district or city.
        place = sem[-1]
        assert place in (top.district, top.city), f"{r['query_id']} place {place} not on top POI"
        checked += 1
    assert checked >= 60, f"expected many attribute/category rows to re-verify, got {checked}"


# --------------------------------------------------------------------------- #
# degradation + Vietnamese integrity
# --------------------------------------------------------------------------- #
def test_degradation_sanity(rows):
    """>= 30% of queries carry a text degradation detectable from the artifact alone: either the
    text has NO diacritics (stripped) or a Typo flag is recorded in skills_tested."""
    degraded = sum(
        1 for r in rows if (not _has_diacritic(r["input_query"])) or ("Typo" in r["skills_tested"])
    )
    assert degraded / len(rows) >= 0.30, f"only {degraded}/{len(rows)} show a stripped/typo degradation"


def test_diacritics_preserved_in_non_degraded(rows):
    """Non-degraded queries (no degradation flag in skills_tested) keep Vietnamese diacritics."""
    non_degraded = [r for r in rows if _DEGRADE_FLAGS.isdisjoint(r["skills_tested"])]
    assert non_degraded, "expected some non-degraded queries"
    with_diac = sum(_has_diacritic(r["input_query"]) for r in non_degraded)
    assert with_diac / len(non_degraded) >= 0.90


def test_no_verbatim_official_query(rows):
    """No generated query reuses an official eval query verbatim (integrity / no eval fitting)."""
    official = {q.input_query for q in load_eval()}
    offenders = [r["query_id"] for r in rows if r["input_query"] in official]
    assert not offenders, f"verbatim official queries: {offenders}"


def test_families_and_query_categories_use_official_vocab(rows, records):
    """query_category values stay within the official Public_Evaluation vocabulary so downstream
    reporting groups cleanly; degradation mix spans several styles."""
    official_qcats = {q.query_category for q in load_eval()}
    assert {r["query_category"] for r in rows} <= official_qcats
    assert len({rec.family for rec in records}) >= 5
    assert len({rec.style for rec in records}) >= 4
